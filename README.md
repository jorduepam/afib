# AFib

AFib is an open-source package which contains all the Python, Jupyter Notebook and C scripts that were used for the following publication: "Influence of the flow split ratio on the position of the main atrial vortex implications for stasis on the left atrial appendage" by Sergio Rodríguez-Aparicio et al., which is currently under review. 

This package includes the implementations of the three routines described in the Supplementary Material: the implementation of a smooth atrial cinematic model (Algorithm 1), the implementation of universal atrial coordinates (Algorithm 2), and the script to calculate the moments of blood age (Algorithm 3). AFib is written in Python, Jupyter Notebook, and C.

## Contents:

- **atrial_cinematic_model**: a python script which contains the functions to generate the smooth atrial wall motion used in the Manuscript, together with a Jupyter Notebook for ease of use.
- **universal_atrial_coordinates**: a Python script which contains the functions to calculate the universal atrial coordinates used in the Manuscript, together with a Jupyter Notebook for ease of use.
- **stagnant_volume_calculation**: a C script to use together with ANSYS Fluent to calculate the moments of blood age. 


## Usage:

The AFib package is a comprehensive collection of scripts, so we refer to the different Jupyter Notebooks which illustrate prototypical use cases of the toolbox.

In the **atrial_cinematic_model** subdirectory, we provide a sample geometry to recreate the wall motion in a cylinder. 

## References:

If you use this software for a publication, please cite these papers:

**[1]** A comprehensive comparison of various patient-specific CFD models of the left atrium for atrial fibrillation patients. J. Dueñas-Pamplona et al. 2021;133:104423. https://doi.org/10.1016/j.compbiomed.2021.104423

**[2]** Reduced-order models of endocardial shear stress patterns in the left atrial appendage from a data-augmented patient-specific database. J. Dueñas-Pamplona et al. 2024;130:713-727. https://doi.org/10.1016/j.apm.2024.03.027

**[3]** Influence of the flow split ratio on the position of the main atrial vortex implications for stasis on the left atrial appendage. S. Rodríguez-Aparicio et al. Currently under review. 

[1] introduces the stagnant volume calculation for the left atrial appendage, while [2] focuses on the parameterization of the atrium by universal coordinates, and [3] join these techniques with the implementation of a smooth atrial motion.

For the calculation of the stagnant volume, please also cite:

[4] Spatial distribution of mean age and higher moments of unsteady and reactive tracers: Reconstruction of residence time distributions. J. Sierra-Pallares et al. 2017;46:312-327. https://doi.org/10.1016/j.apm.2017.01.054
