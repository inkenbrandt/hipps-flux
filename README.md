# hipps-flux <!-- omit in toc -->

*Python utilities for calculating, correcting, and exploring turbulent‐flux data – inspired by the work of Lawrence E. Hipps.*

[![License: GPL-3.0](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/hipps-flux.svg)](https://pypi.org/project/hipps-flux/)
[![Tests](https://github.com/inkenbrandt/hipps-flux/actions/workflows/ci.yml/badge.svg)](https://github.com/inkenbrandt/hipps-flux/actions)
[![Docs](https://readthedocs.org/projects/hipps-flux/badge/?version=latest)](https://hipps-flux.readthedocs.io)

`hipps-flux` is a lightweight, **GPL-3.0** licensed Python package that aims to make common
eddy-covariance and surface-energy-balance calculations *transparent, reproducible, and easy to
extend*.  Core goals include:

* **Straight-forward flux calculations** – sensible, latent, soil and net-radiation helpers.  
* **Energy-balance closure diagnostics & corrections** – EBR, Bowen-ratio, and regression methods.  
* **Metadata aware I/O** – read/write AmeriFlux style files or Pandas/Xarray datasets painlessly.  
* **Plotting shortcuts** – quick-look functions for diurnal composites, closure scatter plots, etc.  
* **First-class documentation & tests** – Sphinx docs (built on ReadTheDocs) and a growing pytest suite.

> **Why “Hipps”?**  
> The name pays homage to Dr. Larry Hipps, whose research laid much of the groundwork for modern
> surface-energy-balance and advective-flux studies. While the codebase is brand new, many of the
> equations and recommended workflows trace their lineage to his publications.

---

## Table of Contents <!-- omit in toc -->
1. [Quick start](#quick-start)
2. [Installation](#installation)
3. [Example usage](#example-usage)
4. [Project layout](#project-layout)
5. [Contributing](#contributing)
6. [Citing](#citing)
7. [License](#license)

---

## Quick start
```bash
pip install hipps-flux                 # from PyPI (recommended)
# or
pip install git+https://github.com/inkenbrandt/hipps-flux.git@main
