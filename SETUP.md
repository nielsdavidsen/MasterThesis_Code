# Setup Guide

This is a fork of [CellModeller](https://github.com/cellmodeller/CellModeller) used for the thesis project.
It explains how to get a working environment after cloning (or forking) this repo — for Maria, our
supervisor, or anyone else picking this up.

## Prerequisites

- **Git**
- **A conda-family package manager**: [miniforge](https://github.com/conda-forge/miniforge) (installs
  both `conda` and `mamba`) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
  are the easiest options. Plain `conda` (Anaconda/Miniconda) also works — just swap `mamba` for `conda`
  in the commands below.
- **An OpenCL runtime** — simulations run on the GPU/CPU via OpenCL, so you need a working OpenCL
  installation for your platform:
  - **macOS**: nothing to install, OpenCL ships with the OS.
  - **Linux/Windows**: install your GPU vendor's driver (NVIDIA/AMD/Intel), or
    [`pocl`](http://portablecl.org/) for a CPU-only fallback.

## 1. Clone the repo

```bash
git clone https://github.com/nielsdavidsen/MasterThesis_Code.git
cd MasterThesis_Code
```

(If you forked it, clone your own fork's URL instead.)

## 2. Create the environment

[`environment.yml`](environment.yml) pins the parts that actually need to come from conda-forge
(Python itself, plus `pip`/`setuptools`) — everything else is pulled in via `pip` in the next step.

```bash
mamba env create -f environment.yml
```

This creates an environment named `cellmodeller-thesis` (the name is set in `environment.yml`).

## 3. Activate it

```bash
mamba activate cellmodeller-thesis
```

## 4. Install CellModeller into the environment

From the repo root, with the environment active:

```bash
pip install -e .
```

The `-e` (editable) install means changes you make to the code under `CellModeller/` take effect
immediately, without reinstalling. This also pulls in the remaining dependencies listed in
[`setup.py`](setup.py) — `numpy`, `scipy`, `pyopengl`, `mako`, `pyqt5`, `pyopencl`, `reportlab`,
`matplotlib` — from PyPI.

## 5. Verify it works

Run one of the bundled example models through the GUI:

```bash
python Scripts/CellModellerGUI.py Examples/ex1_simpleGrowth.py
```

A window should open and start simulating cell growth. If it does, you're set up correctly.

## Repo layout

- `CellModeller/` — the package itself (biophysics, signalling, regulation, integration, GUI).
- `Scripts/` — entry points: the GUI (`CellModellerGUI.py`), headless batch runners, analysis tools.
- `Examples/` — starter model scripts, including step-by-step tutorials (`Tutorial_1`, `Tutorial_2`, ...).
- `environment.yml` / `setup.py` — environment and package definition (this guide covers both).

## Troubleshooting

- **This repo requires Python 3.12.** Earlier commits used Python 3.10, but `setuptools` removed the
  `pkg_resources` module that older CellModeller code depended on, and the Python 3.10 `scipy` wheels
  crash on recent macOS versions. Both are fixed on `master`/`niels`/`maria` as of the Python 3.12
  migration — if you're on an older commit or a different fork, you may hit either issue.
- **`pyopencl` can't find a platform/device**: check `python -c "import pyopencl as cl; print(cl.get_platforms())"`
  lists something. If it's empty, your OpenCL runtime isn't installed/visible (see Prerequisites above).
- **Updating an existing environment** after `environment.yml` changes:
  ```bash
  mamba env update -f environment.yml --prune
  ```
  If that gets messy, it's usually simplest to delete and recreate:
  ```bash
  mamba env remove -n cellmodeller-thesis
  mamba env create -f environment.yml
  ```
