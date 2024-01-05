# reproducibility-alignn

This repo contains the code for the paper *"A reproducibility study of atomistic line graph neural networks for materials property prediction"* by K. Li *et al* [Link to add]. This study aims to reproduce the [work](https://www.nature.com/articles/s41524-021-00650-1) conducted by K. Choudhary and B. DeCost, which centered on the development of Atomistic Line Graph Neural Network (ALIGNN) models for improved prediction of materials properties.

The following files are included in this repo:
- *setup_env.bash*: the bash script to set up the virtual python environment.
- *reproduce_jdft2021.py*: the python script to retrieve the data and create a folder for each property with the associated id_prop.csv file and the POSCAR file for each entry. The script can be run by the command (provided that the *config.json* file exists)
```
python reproduce_jdft2021.py --target $target --random_state $random_state
```
- *run_model_performance.bash*: the bash script to run *reproduce_jdft2021.py* for all the properties considered in the reproducibility study.
- *run_ablation.bash*: the bash script to run *reproduce_jdft2021.py* for the ablation analysis considered in the reproducibility study.
- *plot.py*: the python script to generate the plot using the output data (see the Zenodo link below for the output data)

**The generated output data and the python script for the figures and tables in the reproducibility paper can be found on [Zenodo](https://zenodo.org/records/10042543).**
