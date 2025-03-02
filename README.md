# PolymerTC
## Table of Contents
1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Requirements](#software-requirements)
    1. [OS Requirements](#os-requirements)
    2. [Software Package Dependencies](#software-package-dependencies)
    3. [Python Dependencies](#python-dependencies)
4. [Installation Guide](#installation-guide)
5. [Demo](#demo)
6. [Instructions for Use](#instructions-for-use)
    1. [Step 1. Generate Morphology](#1-generate-morphology)
    2. [Step 2. Calculate Thermal Conductivity](#2-calculate-thermal-conductivity)
    3. [Step 3. Calculate Dynamic Structure Factors](#3-calculate-dynamic-structure-factors)
    4. [Step 4. Fit Dynamic Structure Factors](#4-fit-dynamic-structure-factors)
    5. [Step 5. Estimate Propagons Thermal Conductivity](#5-estimate-propagons-thermal-conductivity)
    6. [Step 6. Calculate Phonon Band and Longitudinality](#6-calculate-phonon-band-and-longitudinality)

## Overview
`PolymerTC` includes code and workflows for studying vibrational thermal transport in polymer semiconductors, integrating molecular dynamics (MD) simulations with density functional theory (DFT) calculations. It is divided into six sections, providing either input files for existing open-source codes or sample codes for data analysis, along with the expected outputs.

## Hardware requirements
`PolymerTC` requires only a standard computer with sufficient RAM to support in-memory operations.

## Software requirements
### OS Requirements
These codes are intended to be supported on both *Linux* and *Windows* platforms and have been tested on
+ Linux x86_64

### Software Package Dependencies
+ [LAMMPS v23Jun2022](https://www.lammps.org/download.html)
+ [GROMACS v2022.2](https://www.gromacs.org/Downloads)

### Python Dependencies
+ python >= 3.7
+ numpy >= 1.24.1
+ scipy >= 1.10.1
+ matplotlib >= 3.7.1
+ dynasor == 2.0
+ phonopy >= 2.21.1
+ GRO2LAM == 1.25

## Installation Guide
No additional installation is needed to use `PolymerTC` as long as all dependencies have been successfully installed. Then:
```
git clone https://github.com/chunlinxuizz/PolymerTC
cd PolymerTC
```

## Demo
Currently, `PolymerTC` consists six components, which include
+ Generate morphology
+ Calculate thermal conductivity
+ Calculate dynamic structure factor
+ Fit dynamic structure factor
+ Estimate propagons thermal conductivity
+ Calculate phonon band and longitudinality

Each component is housed in its own folder and contains either input files for existing open-source codes or sample codes for data analysis, along with the expected outputs.

## Instructions for Use
### 1. Generate morphology
```
cd step1-generate_morphology
sbatch run.slurm
# Note that the Slurm script should be modified to accommodate your High-Performance Computing (HPC) system.
```
After completing the GROMACS calculations, a geometry file named `npt-equial_pbcatom.gro` will be generated, Use this file, along with the toplogical files located in the `force_fields` folder, to generate LAMMPS data files using the [`GRO2LAM`](https://github.com/hernanchavezthielemann/GRO2LAM) code. The expected runtime for this example is approximately 5 hours on an *x86-64, 2.5 GHz, 64-cores* computer.

### 2. Calculate thermal conductivity
```
cd step2-calculate_thermal_conductivity
# Run non-equilibrium MD (NEMD) simulations
sbatch lmp.slurm
# Note that the slurm file should be modified to fit your HPC system
# After finishing the LAMMPS calculations
python kappa.py
```
The `kappa.py` script fits the NEMD calculated temperature gradients and heat currents to compute thermal conductivities. The expected runtime for this example is approximately 6 hours on an *x86-64, 2.5 GHz, 64-cores* computer.

### 3. Calculate dynamic structure factors
```
cd step3-calculate_dynamic_structure_factors
sbatch dynasor.slurm
# Note that the slurm file should be modified to fit your HPC system
```
Before running this step, replace the file named `lammps_trajectory_reader.py` located in [`/dynasor/trajectory/`](https://gitlab.com/materials-modeling/dynasor) with the modefied version in the `step3-calculate_dynamic_structure_factors` folder to enable mass weighting. A total of 20 equilibrium MD simulations will be carried out, each with a duration of 40 ps. Atomic velocities and positions will be recorded throughout the simulations. The dynamic structure factors for each trajectory will be computed using the  [`dynasor`](https://gitlab.com/materials-modeling/dynasor) code, and finally, the average dynamic structure factors will be saved in the `output.pickle` file. The expected runtime for this example is approximately 45 hours on an *x86-64, 2.5 GHz, 64-cores* computer.
### 4. Fit dynamic structure factors
```
cd step4-fit_dynamic_structure_factors
python dsf_fitter.py
```
`dsf_fitter.py` is a Python code designed to automatically fit the dynamic structure factors calculated in Step 3 with damped harmonic oscillators (DHOs). The 
peak frequencies and mode linewidthes will be displayed on the secreen. 
However, it is crucial to note that the parameters within `dsf_fitter.py`, such as the number of peaks, should be manually adjuested to accommodate peaks in different frequency regions. 
### 5. Estimate propagons thermal conductivity
```
cd step5-estimate_propagons_thermal_conductivity
python estimate_propagons_TC.py
```
`estimate_propagons_TC.py` is a Python script used to estimate the propagons-mediated thermal conductivities utilizing data extrated in Step 4. The input files are text files that contain the data obtained by fitting the dynamic_structure_factors described in Step 4, including q-points, peak frequencies, and mode linewidths.
### 6. Calculate phonon band and longitudinality
```
cd step6-calculate_phonon_band_and_longitudinality
unzip FORCE_SETS.zip
phonopy --cp2k -c pbttt.inp -p -s band.conf
bash extract_eigenvectors.sh
python calculate_backbone_longitudinality.py
python calculate_sidechain_longitudinality.py
```
A `band.yaml` file containing phonon eigenvalues and eigenvectors will be generated by the [`Phonopy`](https://phonopy.github.io/phonopy/) package. Note that the `FORCE_SETS` file was calculated by [`Phonopy`](https://phonopy.github.io/phonopy/) based on the outputs of the [`CP2K`](https://www.cp2k.org/) software.  However, this step can also accommodate other software programs that are compatible with [`Phonopy`](https://phonopy.github.io/phonopy/). Subsequently, the data at each q-point in `band.yaml` will be extracted using the `extract_eigenvectors.sh` script. Finally, the backbone and side chain longitudinalitiies will be calculated with `calculate_backbone_longitudinality.py` and `calculate_sidechain_longitudinality.py`, respectively.
