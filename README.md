# PolymerTC

## Overview
`PolymerTC` contains codes and workflow for studing the vibrational thermal transport in polymer semiconductors, combining molecular dynamics (MD) simulations and density functional theory (DFT) calculations. It is devided into six parts, with either the input files for existing open source codes or sample codes for data analysis, as well as expected outputs.

## Hardware requirements
`PolymerTC` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
These codes are expected to be supported for *Windows* and *Linux*, and have been tested on
+ Linux x86_64

### Software Package Dependencies
+ LAMMPS v23Jun2022
+ GROMACS v2022.2

### Python Dependencies
+ numpy
+ scipy
+ matplotlib
+ dynasor
+ phonopy
+ GRO2LAM

## Installation Guide
No additional installation is needed to use `PolymerTC` as long as all dependencies have been successfully installed. Then
```
git clone https://github.com/chunlinxuizz/PolymerTC
cd PolymerTC
```

## Demo
`PolymerTC` current contains six parts, including
+ Generate morphology
+ Calculate thermal conductivity
+ Calculate dynamic structure factor
+ Fit dynamic structure factor
+ Estimate propagons thermal conductivity
+ Calculate phonon band and longitudinality
each part is included in a folder and contains either the input files for existing open source codes or sample codes for data analysis, as well as expected outputs.

## Instructions for Use
### 1. Generate morphology
```
cd step1-generate_morphology
sbatch run.slurm
# Note that the slurm file should modefied to fit your HPC system
```
After finished GROMACS calculations, a geometry file named `npt-equial_pbcatom.gro` will be generated, use it and toplogical files in `force_fields` folder to generate LAMMPS data files with [`GRO2LAM`](https://github.com/hernanchavezthielemann/GRO2LAM) code.
### 2. Calculate thermal conductivity
```
cd step2-calculate_thermal_conductivity
# Run non-equilibrium MD (NEMD) simulations
sbatch lmp.slurm
# Note that the slurm file should modefied to fit your HPC system
# After finished the LAMMPS calculations
python kappa.py
```
The `kappa.py` fits the NEMD calculated temperature gradients and heat currents to compute thermal conductivities.

### 3. Calculate dynamic structure factors
```
cd step3-calculate_dynamic_structure_factors
# Replace the file /dynasor/trajectory/lammps_trajectory_reader.py with the file in this fold to enable mass weighting
sbatch dynasor.slurm
```
Successive equilibrium MD simulations will be conducted, and the atomic velocities and positions will be saved. Dynamic structure factors of each period of time are calculated with [`dynasor`](https://gitlab.com/materials-modeling/dynasor) code, and eventually, the averaged dynamic structure factors will be saved in `output.pickle`. 
### 4. Fit dynamic structure factors
```
cd step4-fit_dynamic_structure_factors
python dsf_fitter.py
```
`dsf_fitter.py` is a python code to automatically fit the dynamic structure factors calculated in step 3. The 
peak frequencies and mode linewidthes will be outputted in the secreen. 
However, it is important to note that the parameters in `dsf_fitter.py`, such as the number of peaks, should be manually modefied to fit peaks in different frequency regions.
### 5. Estimate propagons thermal conductivity
```
cd step5-estimate_propagons_thermal_conductivity
python estimate_propagons_TC.py
```
`estimate_propagons_TC.py` is a python code to estimate propagons-mediated thermal conductivities using data extrated in step 4. The input files are text files containing q_points, peak frequencies, and mode linewidths.
### 6. Calculate phonon band and longitudinality
```
cd step6-calculate_phonon_band_and_longitudinality
phonopy --cp2k -c pbttt.inp -p -s band.conf
./extract_eigenvectors.sh
python calculate_backbone_longitudinality.py
python calculate_sidechain_longitudinality.py
```
A `band.yaml` file contains phonon eigenvalues and eigenvectors will be generated by [`phonopy`](https://phonopy.github.io/phonopy/). Then the data at each q point in `band.yaml` will be extracted by `extract_eigenvectors.sh`. Eventually, the backbona and side chain longitudinalitiies will be calculated with `calculate_backbone_longitudinality.py` and `calculate_sidechain_longitudinality.py`, respectively.
