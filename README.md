# Honeybee Nest-Site Selection

[![Binder](https://gesis.mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/v2/gh/daniellechase/honeybee-nestsiteselection/main?urlpath=voila%2Frender%2Fhoneybee_demo.ipynb)

Interactive simulation of honeybee nest-site selection, based on Seeley et al. (2012). Includes an agent-based model (ABM), ODE mean-field approximation, phase portrait, and bifurcation diagram.

## Run in browser

Click the Binder badge above — no installation required. The app may take a minute to load.

## Run locally

```bash
git clone https://github.com/daniellechase/honeybee-nestsiteselection.git
cd honeybee-nestsiteselection
pip install numpy matplotlib scipy
python honeybee_sim.py
```

## Using the simulation

The display shows three panels:
- **Left** — animated ABM: grey = uncommitted, blue = site A, yellow = site B
- **Top right** — fraction of bees in each state over time
- **Bottom right** — phase portrait (φ_A vs φ_B) with fixed points and the current ABM trajectory

### Controls

**Sliders** — adjust the four rate parameters for each site:
- **Commit** (α^c): rate at which uncommitted bees begin scouting a site
- **Recruit** (α^r): rate at which committed bees recruit others via waggle dance
- **Abandon** (α^a): rate at which committed bees stop scouting
- **Stop signal** (α^s): rate at which bees from one site inhibit the other (piping behavior)

Each slider has a text box on the right where you can type a value directly.

**N bees** slider — changes the number of bees in the ABM.

**Equal sites** checkbox — when checked, Site A and Site B sliders are linked and move together (symmetric competition). Uncheck it to set Site A and Site B parameters independently and explore asymmetric scenarios.

**Reset** — restarts the simulation with the current parameter values.

**Pause / Resume** — freezes and unfreezes the animation.

**Bifurcation** — opens a bifurcation diagram showing stable and unstable fixed points as the stop-signal rate σ varies, with a marker at the current σ value.
