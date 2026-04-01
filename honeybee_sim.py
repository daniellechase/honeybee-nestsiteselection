"""
Honeybee Nest-Site Selection Simulation
========================================
Model: Seeley, Visscher, Schlegel, Hogan, Franks & Marshall (2012), Science 335:108-11.

States:  U = uncommitted,  A = committed to site A,  B = committed to site B

ODE (mean-field, Eq. 1-2 in Chase & Peleg 2025):
  dφ_A/dt = α_A^c(1−φ_A−φ_B) + α_A^r φ_A(1−φ_A−φ_B) − α_A^a φ_A − α_B^s φ_A φ_B
  dφ_B/dt = α_B^c(1−φ_A−φ_B) + α_B^r φ_B(1−φ_A−φ_B) − α_B^a φ_B − α_A^s φ_A φ_B
"""

import numpy as np
import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button, TextBox, CheckButtons
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# ── Default parameters ─────────────────────────────────────────────────────────
N    = 100
AC_A = 3;   AC_B = 3
AR_A = 3;   AR_B = 3
AA_A = 1/3; AA_B = 1/3
AS_A = 10;  AS_B = 10
DT   = 0.06

# ── Colours ────────────────────────────────────────────────────────────────────
BG     = "white"
PLOTBG = "#f0f4ff"
EDGE   = "#c0c8e0"
TEXT   = "#222244"
COL_A  = "#6ca3e7"
COL_B  = "#f9bb41"
COL_U  = "#999999"

# ── ODE ────────────────────────────────────────────────────────────────────────
def ode_rhs(t, y, ac_a, ac_b, ar_a, ar_b, aa_a, aa_b, as_a, as_b):
    phi_a, phi_b = y
    phi_u = max(1.0 - phi_a - phi_b, 0.0)
    da = ac_a * phi_u - aa_a * phi_a + ar_a * phi_a * phi_u - as_b * phi_a * phi_b
    db = ac_b * phi_u - aa_b * phi_b + ar_b * phi_b * phi_u - as_a * phi_a * phi_b
    return [da, db]

def ode_rhs_vec(PA, PB, params):
    ac_a, ac_b, ar_a, ar_b, aa_a, aa_b, as_a, as_b = params
    PU = np.maximum(1.0 - PA - PB, 0.0)
    dA = ac_a * PU - aa_a * PA + ar_a * PA * PU - as_b * PA * PB
    dB = ac_b * PU - aa_b * PB + ar_b * PB * PU - as_a * PA * PB
    return dA, dB

ODE_HOR = 100.0

def _shared_ic(N):
    """Random IC: 0 or 1 bee committed to each site."""
    rng = np.random.default_rng()
    na = int(rng.integers(0, 2))
    nb = int(rng.integers(0, 2))
    return [na / N, nb / N], na, nb

def solve_ode(params, ic, horizon=ODE_HOR):
    sol = solve_ivp(ode_rhs, [0, horizon], list(ic), args=params,
                    t_eval=np.linspace(0, horizon, 800),
                    method="RK45", rtol=1e-8)
    phi_a, phi_b = sol.y
    phi_u = np.maximum(1.0 - phi_a - phi_b, 0.0)
    return sol.t, phi_u, phi_a, phi_b

def bifurcation_diagram(params, sigma_vals):
    """Sweep σ (AS_A = AS_B), find all fixed points robustly."""
    ac_a, ac_b, ar_a, ar_b, aa_a, aa_b, _, _ = params
    gamma = (ac_a + ac_b) / 2
    rho   = (ar_a + ar_b) / 2
    alpha = (aa_a + aa_b) / 2

    sym_rows  = []   # (sig, phi*, stable)
    asym_rows = []   # (sig, phi_hi, phi_lo, stable)

    def residual(y, sig):
        return ode_rhs(0, y, ac_a, ac_b, ar_a, ar_b, aa_a, aa_b, sig, sig)

    def stability(fp, sig):
        eps = 1e-6
        f0  = np.array(residual(fp, sig))
        J   = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2); e[j] = eps
            J[:, j] = (np.array(residual(fp + e, sig)) - f0) / eps
        return bool(np.all(np.real(np.linalg.eigvals(J)) < 0))

    # ICs targeting asymmetric (consensus) fixed points
    asym_ics = [[0.85, 0.01], [0.01, 0.85],
                [0.70, 0.03], [0.03, 0.70],
                [0.55, 0.05], [0.05, 0.55],
                [0.90, 0.05], [0.05, 0.90]]

    for sig in sigma_vals:
        # ── Symmetric FP: analytic quadratic ──────────────────────────────
        A_ = 2*rho + sig
        B_ = 2*gamma + alpha - rho
        disc = B_**2 + 4*A_*gamma
        if A_ > 0 and disc >= 0:
            phi_s = (-B_ + np.sqrt(disc)) / (2 * A_)
            if 0.0 < phi_s < 0.5:
                lam2 = rho * (1 - 2*phi_s) - alpha   # anti-sym eigenvalue
                sym_rows.append((sig, phi_s, lam2 < 0))

        # ── Asymmetric FPs: targeted numerical search ──────────────────────
        found = []
        for ic0 in asym_ics:
            try:
                fp, _, ier, _ = fsolve(residual, ic0, args=(sig,),
                                       full_output=True)[:4]
                if ier != 1: continue
                fp = np.array(fp)
                if fp[0] < -0.01 or fp[1] < -0.01 or fp.sum() > 1.01: continue
                fp = np.clip(fp, 0.0, 1.0)
                if np.max(np.abs(residual(fp, sig))) > 1e-7: continue
                if abs(fp[0] - fp[1]) < 0.05: continue   # skip symmetric
                if all(np.linalg.norm(fp - u) > 0.02 for u in found):
                    found.append(fp)
            except Exception:
                pass
        for fp in found:
            hi, lo = (fp[0], fp[1]) if fp[0] > fp[1] else (fp[1], fp[0])
            asym_rows.append((sig, hi, lo, stability(fp, sig)))

    sym_arr  = np.array(sym_rows)  if sym_rows  else np.zeros((0, 3))
    asym_arr = np.array(asym_rows) if asym_rows else np.zeros((0, 4))
    return sym_arr, asym_arr

def find_fixed_points(params):
    def f(y):
        return ode_rhs(0, y, *params)
    candidates = []
    grid = np.linspace(0.01, 0.97, 12)
    for pa0 in grid:
        for pb0 in grid:
            if pa0 + pb0 >= 0.99:
                continue
            try:
                fp, _, ier, _ = fsolve(f, [pa0, pb0], full_output=True)[:4]
                if ier != 1: continue
                if fp[0] < -0.02 or fp[1] < -0.02 or fp[0] + fp[1] > 1.02: continue
                fp = np.clip(fp, 0.0, 1.0)
                if np.max(np.abs(f(fp))) > 1e-7: continue
                candidates.append(fp)
            except Exception:
                pass
    unique = []
    for fp in candidates:
        if all(np.linalg.norm(fp - u) > 0.02 for u in unique):
            unique.append(fp)
    classified = []
    eps = 1e-6
    for fp in unique:
        f0 = np.array(f(fp))
        J = np.zeros((2, 2))
        for j in range(2):
            e = np.zeros(2); e[j] = eps
            J[:, j] = (np.array(f(fp + e)) - f0) / eps
        eigvals = np.linalg.eigvals(J)
        classified.append((fp, bool(np.all(np.real(eigvals) < 0))))
    return classified

def build_phase_data(params, n_stream=60):
    v = np.linspace(0.0, 1.0, n_stream)
    PA, PB = np.meshgrid(v, v)
    valid = (PA + PB) < 1.0
    dA, dB = ode_rhs_vec(PA, PB, params)
    dA = np.where(valid, dA, 0.0)
    dB = np.where(valid, dB, 0.0)
    fps = find_fixed_points(params)
    return (v, dA, dB, fps)

# ── Agent-based model ──────────────────────────────────────────────────────────
class BeeABM:
    COLORS = {0: COL_U, 1: COL_A, 2: COL_B}

    def __init__(self, N, params, dt):
        self.N = N; self.dt = dt
        self.rng = np.random.default_rng()
        self._unpack(params)
        self.reset()

    def _unpack(self, p):
        (self.ac_a, self.ac_b, self.ar_a, self.ar_b,
         self.aa_a, self.aa_b, self.as_a, self.as_b) = p

    def reset(self):
        self.rng = np.random.default_rng()
        rng = self.rng
        self.state = np.zeros(self.N, dtype=np.int8)
        theta = rng.uniform(0, 2 * np.pi, self.N)
        r     = rng.uniform(0.05, 0.44,   self.N)
        self.x = 0.5 + r * np.cos(theta)
        self.y = 0.5 + r * np.sin(theta)
        self.t = 0.0
        self.hist_t = [0.0]
        self.hist   = {s: [float(s == 0)] for s in (0, 1, 2)}

    def step(self):
        dt = self.dt; N = self.N
        phi_a = np.sum(self.state == 1) / N
        phi_b = np.sum(self.state == 2) / N
        rng  = self.rng
        rand = rng.random(N)
        new  = self.state.copy()
        for i in range(N):
            s = self.state[i]; r = rand[i]
            if s == 0:
                p_a = min((self.ac_a + self.ar_a * phi_a) * dt, 1.0)
                p_b = min((self.ac_b + self.ar_b * phi_b) * dt, 1.0 - p_a)
                if   r < p_a:        new[i] = 1
                elif r < p_a + p_b:  new[i] = 2
            elif s == 1:
                if r < min((self.aa_a + self.as_b * phi_b) * dt, 1.0): new[i] = 0
            else:
                if r < min((self.aa_b + self.as_a * phi_a) * dt, 1.0): new[i] = 0
        self.state = new; self.t += dt
        self.x += self.rng.normal(0, 0.005, N)
        self.y += self.rng.normal(0, 0.005, N)
        self.x  = np.clip(self.x, 0.02, 0.98)
        self.y  = np.clip(self.y, 0.02, 0.98)
        counts = {s: np.sum(self.state == s) / N for s in (0, 1, 2)}
        self.hist_t.append(self.t)
        for s in (0, 1, 2): self.hist[s].append(counts[s])

    def colors(self):
        return [self.COLORS[s] for s in self.state]

# ── Initial data ───────────────────────────────────────────────────────────────
params0 = (AC_A, AC_B, AR_A, AR_B, AA_A, AA_B, AS_A, AS_B)
abm     = BeeABM(N, params0, DT)
phase_data = build_phase_data(params0)

# ── Figure layout: 2 rows × 2 cols ─────────────────────────────────────────────
# Col 0: bee arena (spans both rows)
# Col 1 row 0: ABM + ODE time series (merged)
# Col 1 row 1: phase portrait
fig = plt.figure(figsize=(16, 9), facecolor=BG)
fig.canvas.manager.set_window_title("Honeybee Nest-Site Selection")

gs = gridspec.GridSpec(
    2, 2, figure=fig,
    left=0.04, right=0.97, top=0.92, bottom=0.30,
    hspace=0.42, wspace=0.30,
    width_ratios=[1.1, 1.0],
)
ax_bee   = fig.add_subplot(gs[:, 0])
ax_abm   = fig.add_subplot(gs[0, 1])
ax_phase = fig.add_subplot(gs[1, 1])

for ax in (ax_bee, ax_abm, ax_phase):
    ax.set_facecolor(PLOTBG)
    for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
    ax.tick_params(colors=TEXT, labelsize=7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)

# ── Bee arena ──────────────────────────────────────────────────────────────────
ax_bee.set_xlim(0, 1); ax_bee.set_ylim(0, 1)
ax_bee.set_aspect("equal"); ax_bee.axis("off")
scat = ax_bee.scatter(abm.x, abm.y, c=abm.colors(), s=55, marker="h",
                      edgecolors=EDGE, linewidths=0.3, zorder=3)
_lbl_kw = dict(transform=ax_bee.transAxes, fontsize=8, va="bottom")
info_u = ax_bee.text(0.02, 0.14, "", color=COL_U, **_lbl_kw)
info_a = ax_bee.text(0.02, 0.09, "", color=COL_A, **_lbl_kw)
info_b = ax_bee.text(0.02, 0.04, "", color=COL_B, **_lbl_kw)
for _t in (info_u, info_a, info_b):
    _t.set_bbox(dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.55, edgecolor="none"))

# ── ABM time-series panel ──────────────────────────────────────────────────────
ax_abm.set_xlim(0, ODE_HOR); ax_abm.set_ylim(-0.02, 1.02)
ax_abm.set_title("ABM", fontsize=9)
ax_abm.set_xlabel("Time"); ax_abm.set_ylabel("Fraction")
ax_abm.grid(color=EDGE, lw=0.5, alpha=0.8)
l_u, = ax_abm.plot([], [], color=COL_U, lw=1.7)
l_a, = ax_abm.plot([], [], color=COL_A, lw=1.7)
l_b, = ax_abm.plot([], [], color=COL_B, lw=1.7)

# ── Phase portrait panel ───────────────────────────────────────────────────────
ax_phase.set_xlim(-0.02, 1.02); ax_phase.set_ylim(-0.02, 1.02)
ax_phase.set_aspect("equal")
ax_phase.set_title(r"Phase portrait  ($\phi_A$ vs $\phi_B$)", fontsize=9)
ax_phase.set_xlabel(r"$\phi_A$")
ax_phase.set_ylabel(r"$\phi_B$")
ax_phase.grid(color=EDGE, lw=0.4, alpha=0.6)
ax_phase.fill_between([0, 1], [1, 0], [1, 1], color="#e8e8e8", zorder=0)
ax_phase.plot([0, 1], [1, 0], color=EDGE, lw=1.0, zorder=1)

phase_artists = []

def draw_phase(pd, ax):
    (v, dA, dB, fps) = pd
    arts = []
    strm = ax.streamplot(v, v, dA, dB, color="#555555", linewidth=0.7,
                          density=1.8, arrowsize=1.0, zorder=2)
    arts.append(strm)
    stable_plotted = False; unstable_plotted = False
    for (fp, stable) in fps:
        if stable:
            h, = ax.plot(fp[0], fp[1], 'o', color="#111111", ms=8,
                         mfc="#111111", mec="#111111", zorder=8)
            stable_plotted = True
        else:
            h, = ax.plot(fp[0], fp[1], 'o', color="#111111", ms=8,
                         mfc="none", mec="#111111", mew=1.8, zorder=8)
            unstable_plotted = True
        arts.append(h)
    handles = [Line2D([0], [0], marker='o', color=COL_U, ms=6, lw=0)]
    labels  = ["ABM state"]
    if stable_plotted:
        handles.append(Line2D([0], [0], marker='o', color='#111111', ms=6, lw=0, mfc='#111111'))
        labels.append("stable FP")
    if unstable_plotted:
        handles.append(Line2D([0], [0], marker='o', color='#111111', ms=6, lw=0, mfc='none', mew=1.5))
        labels.append("unstable FP")
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=6.5,
                  facecolor="white", labelcolor=TEXT, edgecolor=EDGE)
    return arts

phase_artists.extend(draw_phase(phase_data, ax_phase))
abm_dot_ref = [ax_phase.plot([], [], 'o', color=COL_U, ms=7, zorder=6)[0]]

time_lbl = fig.text(0.5, 0.945, "t = 0.00", ha="center",
                    color=TEXT, fontsize=12, fontweight="bold")

# ── Equal-sites checkbox ───────────────────────────────────────────────────────
equal_mode = [True]   # start in equal mode

ax_chk = fig.add_axes([0.705, 0.058, 0.11, 0.038], facecolor=BG)
chk = CheckButtons(ax_chk, ["Equal sites"], actives=[True])
chk.labels[0].set_color(TEXT); chk.labels[0].set_fontsize(8)

# Pairs: changing one A slider mirrors to its B partner and vice-versa
PAIRS = {
    r"$\alpha_A^c$": r"$\alpha_B^c$",  r"$\alpha_B^c$": r"$\alpha_A^c$",
    r"$\alpha_A^r$": r"$\alpha_B^r$",  r"$\alpha_B^r$": r"$\alpha_A^r$",
    r"$\alpha_A^a$": r"$\alpha_B^a$",  r"$\alpha_B^a$": r"$\alpha_A^a$",
    r"$\alpha_A^s$": r"$\alpha_B^s$",  r"$\alpha_B^s$": r"$\alpha_A^s$",
}

def on_equal_toggle(label):
    equal_mode[0] = not equal_mode[0]
    if equal_mode[0]:
        # Snap B sliders to match A values
        for lbl_a, lbl_b in [(r"$\alpha_A^c$", r"$\alpha_B^c$"),
                              (r"$\alpha_A^r$", r"$\alpha_B^r$"),
                              (r"$\alpha_A^a$", r"$\alpha_B^a$"),
                              (r"$\alpha_A^s$", r"$\alpha_B^s$")]:
            _updating[0] = True
            sliders[lbl_b].set_val(sliders[lbl_a].val)
            textboxes[lbl_b][0].set_val(textboxes[lbl_b][3] % sliders[lbl_a].val)
            _updating[0] = False

chk.on_clicked(on_equal_toggle)

# ── Sliders + TextBoxes ────────────────────────────────────────────────────────
for label, xc in [("Commit", 0.14), ("Recruit", 0.37),
                   ("Abandon", 0.60), ("Stop signal", 0.83)]:
    fig.text(xc, 0.228, label, ha="center", color=TEXT, fontsize=8, fontstyle="italic")

fig.text(0.015, 0.208, "Site A", ha="center", va="center",
         color=COL_A, fontsize=8, fontweight="bold", rotation=90)
fig.text(0.015, 0.148, "Site B", ha="center", va="center",
         color=COL_B, fontsize=8, fontweight="bold", rotation=90)

sl_defs = [
    (r"$\alpha_A^c$",       0.05, 0.198, 0.001, 20.0,  AC_A, "%.3f"),
    (r"$\alpha_B^c$",       0.05, 0.138, 0.001, 20.0,  AC_B, "%.3f"),
    (r"$\alpha_A^r$",       0.28, 0.198, 0.001, 20.0,  AR_A, "%.3f"),
    (r"$\alpha_B^r$",       0.28, 0.138, 0.001, 20.0,  AR_B, "%.3f"),
    (r"$\alpha_A^a$",       0.51, 0.198, 0.001,  5.0,  AA_A, "%.3f"),
    (r"$\alpha_B^a$",       0.51, 0.138, 0.001,  5.0,  AA_B, "%.3f"),
    (r"$\alpha_A^s$",       0.74, 0.198, 0.0,   30.0,  AS_A, "%.2f"),
    (r"$\alpha_B^s$",       0.74, 0.138, 0.0,   30.0,  AS_B, "%.2f"),
    (r"$N_\mathrm{bees}$",  0.28, 0.078, 20,    300,   N,    "%d"),
]

sliders   = {}
textboxes = {}
_updating = [False]

for lbl, lft, yp, vmin, vmax, vi, fmt in sl_defs:
    ax_sl = fig.add_axes([lft, yp, 0.13, 0.022], facecolor="#e8eeff")
    sl = Slider(ax_sl, lbl, vmin, vmax, valinit=vi, color=COL_A, track_color=EDGE)
    sl.label.set_color(TEXT); sl.label.set_fontsize(8)
    sl.valtext.set_visible(False)
    sliders[lbl] = sl

    ax_tb = fig.add_axes([lft + 0.135, yp, 0.042, 0.022], facecolor="white")
    tb = TextBox(ax_tb, "", initial=fmt % vi, textalignment="center")
    tb.text_disp.set_color("black"); tb.text_disp.set_fontsize(7.5)
    for sp in ax_tb.spines.values(): sp.set_edgecolor(EDGE)
    textboxes[lbl] = (tb, vmin, vmax, fmt)

def _make_sl_cb(lbl):
    def cb(val):
        if _updating[0]: return
        _updating[0] = True
        textboxes[lbl][0].set_val(textboxes[lbl][3] % val)
        if equal_mode[0] and lbl in PAIRS:
            partner = PAIRS[lbl]
            sliders[partner].set_val(val)
            textboxes[partner][0].set_val(textboxes[partner][3] % val)
        _updating[0] = False
    return cb

def _make_tb_cb(lbl):
    def cb(text):
        if _updating[0]: return
        _updating[0] = True
        sl = sliders[lbl]; tb, vmin, vmax, fmt = textboxes[lbl]
        try:
            v = float(np.clip(float(text), vmin, vmax))
            sl.set_val(v)
            if equal_mode[0] and lbl in PAIRS:
                partner = PAIRS[lbl]
                sliders[partner].set_val(v)
                textboxes[partner][0].set_val(fmt % v)
        except ValueError:
            tb.set_val(fmt % sl.val)
        _updating[0] = False
    return cb

for lbl in sliders:
    sliders[lbl].on_changed(_make_sl_cb(lbl))
    textboxes[lbl][0].on_submit(_make_tb_cb(lbl))

# ── Buttons ────────────────────────────────────────────────────────────────────
ax_bb = fig.add_axes([0.60, 0.060, 0.10, 0.036], facecolor="white")
ax_bp = fig.add_axes([0.82, 0.060, 0.07, 0.036], facecolor="white")
ax_br = fig.add_axes([0.90, 0.060, 0.07, 0.036], facecolor="white")
btn_bifur = Button(ax_bb, "Bifurcation", color="white", hovercolor="#ddeeff")
btn_pause = Button(ax_bp, "Pause",       color="white", hovercolor="#ddeeff")
btn_reset = Button(ax_br, "Reset",       color="white", hovercolor="#ddeeff")
btn_bifur.label.set_color(TEXT)
btn_pause.label.set_color(TEXT); btn_reset.label.set_color(TEXT)
paused = [False]

def on_pause(event):
    paused[0] = not paused[0]
    btn_pause.label.set_text("Resume" if paused[0] else "Pause")
    fig.canvas.draw_idle()

def on_bifurcation(event):
    p = get_params()
    sigma_vals = np.linspace(0.0, 30.0, 250)
    sym_arr, asym_arr = bifurcation_diagram(p, sigma_vals)
    sig_cur = (p[6] + p[7]) / 2

    bf = plt.figure(figsize=(7, 5), facecolor=BG)
    bf.canvas.manager.set_window_title("Pitchfork Bifurcation")
    ax = bf.add_subplot(111)
    ax.set_facecolor(PLOTBG)
    for sp in ax.spines.values(): sp.set_edgecolor(EDGE)
    ax.tick_params(colors=TEXT)
    ax.xaxis.label.set_color(TEXT); ax.yaxis.label.set_color(TEXT)
    ax.title.set_color(TEXT)
    ax.set_xlabel(r"Stop-signal rate  $\sigma$", fontsize=11)
    ax.set_ylabel(r"Fixed-point fraction  $\phi^*$", fontsize=11)
    ax.set_title("Pitchfork bifurcation  (equal sites)", fontsize=12, color=TEXT)
    ax.set_xlim(sigma_vals[0], sigma_vals[-1])
    ax.set_ylim(-0.02, 1.02)
    ax.grid(color=EDGE, lw=0.5, alpha=0.7)

    def plot_branch(sigmas, values, stabs):
        """Plot a single branch, connected across stable/unstable transitions."""
        order = np.argsort(sigmas)
        s, v, st = sigmas[order], values[order], stabs[order].astype(bool)
        n = len(s)
        i = 0
        while i < n:
            cur = bool(st[i])
            j = i + 1
            while j < n and bool(st[j]) == cur:
                j += 1
            # include one extra point at the end so segments connect
            end = min(j + 1, n)
            ax.plot(s[i:end], v[i:end], color="black", lw=2.0,
                    ls="-" if cur else "--")
            i = j

    # Symmetric branch
    if len(sym_arr):
        plot_branch(sym_arr[:, 0], sym_arr[:, 1], sym_arr[:, 2])

    # Asymmetric branches — prepend the symmetric FP at the bifurcation onset
    if len(asym_arr):
        asym_sigs = asym_arr[:, 0]
        sig_min = asym_sigs.min()
        # find the symmetric FP value at that σ to use as the branch root
        if len(sym_arr):
            closest = np.argmin(np.abs(sym_arr[:, 0] - sig_min))
            root_phi = sym_arr[closest, 1]
            prepend_s  = np.array([sig_min])
            prepend_hi = np.array([root_phi])
            prepend_lo = np.array([root_phi])
            prepend_st = np.array([True])  # connection point styled as stable
        else:
            prepend_s = prepend_hi = prepend_lo = prepend_st = np.array([])

        hi_s  = np.concatenate([prepend_s,  asym_arr[:, 0]])
        hi_v  = np.concatenate([prepend_hi, asym_arr[:, 1]])
        hi_st = np.concatenate([prepend_st, asym_arr[:, 3].astype(bool)])
        lo_v  = np.concatenate([prepend_lo, asym_arr[:, 2]])
        lo_st = hi_st.copy()

        plot_branch(hi_s, hi_v, hi_st)
        plot_branch(hi_s, lo_v, lo_st)

    # Current σ
    ax.axvline(sig_cur, color="#cc3333", lw=1.2, ls=":",
               label=f"current σ = {sig_cur:.2f}")

    from matplotlib.lines import Line2D as L2D
    handles = [
        L2D([0],[0], color="black", lw=2, ls="-",  label="stable FP"),
        L2D([0],[0], color="black", lw=2, ls="--", label="unstable FP"),
        L2D([0],[0], color="#cc3333", lw=1.2, ls=":", label=f"current σ = {sig_cur:.2f}"),
    ]
    ax.legend(handles=handles, fontsize=9, facecolor="white",
              edgecolor=EDGE, labelcolor=TEXT)
    bf.tight_layout()
    bf.show()

btn_bifur.on_clicked(on_bifurcation)
btn_pause.on_clicked(on_pause)

def get_params():
    return (
        sliders[r"$\alpha_A^c$"].val, sliders[r"$\alpha_B^c$"].val,
        sliders[r"$\alpha_A^r$"].val, sliders[r"$\alpha_B^r$"].val,
        sliders[r"$\alpha_A^a$"].val, sliders[r"$\alpha_B^a$"].val,
        sliders[r"$\alpha_A^s$"].val, sliders[r"$\alpha_B^s$"].val,
    )

def on_reset(event):
    p = get_params()
    n = int(sliders[r"$N_\mathrm{bees}$"].val)
    abm.N = n; abm._unpack(p); abm.reset()
    scat.set_offsets(np.c_[abm.x, abm.y])
    scat.set_sizes(np.full(n, 55))
    scat.set_color(abm.colors())

    ax_abm.set_xlim(0, ODE_HOR)
    l_u.set_data([], []); l_a.set_data([], []); l_b.set_data([], [])



    ax_phase.cla()
    ax_phase.set_facecolor(PLOTBG)
    for sp in ax_phase.spines.values(): sp.set_edgecolor(EDGE)
    ax_phase.tick_params(colors=TEXT, labelsize=7)
    ax_phase.set_xlim(-0.02, 1.02); ax_phase.set_ylim(-0.02, 1.02)
    ax_phase.set_aspect("equal")
    ax_phase.set_title(r"Phase portrait  ($\phi_A$ vs $\phi_B$)", fontsize=9, color=TEXT)
    ax_phase.set_xlabel(r"$\phi_A$", color=TEXT)
    ax_phase.set_ylabel(r"$\phi_B$", color=TEXT)
    ax_phase.grid(color=EDGE, lw=0.4, alpha=0.6)
    ax_phase.fill_between([0, 1], [1, 0], [1, 1], color="#e8e8e8", zorder=0)
    ax_phase.plot([0, 1], [1, 0], color=EDGE, lw=1.0, zorder=1)
    phase_artists.clear()
    phase_artists.extend(draw_phase(build_phase_data(p), ax_phase))
    abm_dot_ref[0], = ax_phase.plot([], [], 'o', color=COL_U, ms=7, zorder=6)

    time_lbl.set_text("t = 0.00")
    paused[0] = False
    btn_pause.label.set_text("Pause")
    fig.canvas.draw_idle()

btn_reset.on_clicked(on_reset)

# ── Animation ──────────────────────────────────────────────────────────────────
STEPS = 2

def animate(_frame):
    if paused[0]:
        return scat, l_u, l_a, l_b, abm_dot_ref[0], time_lbl, info_u, info_a, info_b

    for _ in range(STEPS):
        abm.step()

    scat.set_offsets(np.c_[abm.x, abm.y])
    scat.set_color(abm.colors())

    th = np.array(abm.hist_t)
    l_u.set_data(th, abm.hist[0])
    l_a.set_data(th, abm.hist[1])
    l_b.set_data(th, abm.hist[2])
    if abm.t > ax_abm.get_xlim()[1] * 0.9:
        ax_abm.set_xlim(0, ax_abm.get_xlim()[1] * 1.5)

    phi_a_h = np.array(abm.hist[1])
    phi_b_h = np.array(abm.hist[2])
    abm_dot_ref[0].set_data([phi_a_h[-1]], [phi_b_h[-1]])

    time_lbl.set_text(f"t = {abm.t:.1f}")

    N_  = abm.N
    na  = round(abm.hist[1][-1] * N_)
    nb  = round(abm.hist[2][-1] * N_)
    nu  = N_ - na - nb
    info_u.set_text(f"Uncommitted : {nu:3d}")
    info_a.set_text(f"Site A      : {na:3d}")
    info_b.set_text(f"Site B      : {nb:3d}")
    return scat, l_u, l_a, l_b, abm_dot_ref[0], time_lbl, info_u, info_a, info_b

ani = animation.FuncAnimation(fig, animate, frames=None, interval=40, blit=False,
                               cache_frame_data=False)

fig.text(0.5, 0.965, "Honeybee Nest Site Selection",
         ha="center", color=TEXT, fontsize=11, fontweight="bold")

plt.show()
