# Honeybee Nest-Site Selection

[![Binder](https://gesis.mybinder.org/badge_logo.svg)](https://gesis.mybinder.org/v2/gh/daniellechase/honeybee-nestsiteselection/main?urlpath=voila%2Frender%2Fhoneybee_demo.ipynb)

Interactive simulation of honeybee nest-site selection, based on Seeley et al. (2012). The demo was created for a guest lecture in CSCI 5423.

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

Each slider has a text box on the right where you can type a value directly.

**N bees** slider — changes the number of bees in the ABM.

**Equal sites** checkbox — when checked, Site A and Site B sliders are linked and move together. Uncheck it to set Site A and Site B parameters independently.

**Reset** — restarts the simulation with the current parameter values.

**Pause / Resume** — freezes and unfreezes the animation.

**Bifurcation** — opens a bifurcation diagram showing stable and unstable fixed points as the stop-signal rate σ varies, with a marker at the current σ value. Note: this button does not work in the browser version (Binder); run locally to use it.
