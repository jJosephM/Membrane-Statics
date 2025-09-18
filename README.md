# Dataset and Code for: "Three-Dimensional Domain-Wall Membranes"

**Author:** J. J. Mankenberg  
**Affiliations:** Department of Physics and Electrical Engineering, Linnaeus University, SE-39231 Kalmar, Sweden  
**Correspondence:** jamaab@lnu.se  

---

## 📄 Description
This repository contains the plotting and analysis scripts associated with the manuscript:

> J. J. Mankenberg and Ar. Abanov, *Three-Dimensional Domain-Wall Membranes*,  
> submitted to *Physical Review B* (2025).

The data include simulation outputs, processed quantities, and plotting scripts used to generate the figures in the paper.

---

## 📂 Repository Contents
- `scripts/` — Python and CUDA scripts for analysis and figure generation.  
- `README.md` — This file.  
- `LICENSE` — License for reuse of this dataset and code.  

---

## ⚙️ Requirements
The analysis scripts require Python 3.9+, CUDA 12.4+ and the following packages:
- `numpy`
- `scipy`
- `matplotlib`

---

## 🖥️ Computational Resources
Some simulations in this project were performed using:

- **CUDA GPU acceleration** for micromagnetic solvers and energy minimization.  
- **High Performance Research Computing (HPRC) facilities at Texas A&M University** for large-scale runs.  
- Local testing and analysis were performed on NVIDIA GPUs (e.g. RTX 4090).  

For reproducibility, smaller-scale scripts provided in `scripts/` can be run on a local GPU-enabled machine.  
