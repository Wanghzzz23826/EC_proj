
# EC_proj

This repository contains the BrainPy-based implementation and running scripts for the V1 / VCD-related experiments.

## Main Script

The main running script is:

run_vcd_4rec.py

This script is the primary entry point for running the Temporal-LGN + 4-receptor-background VCD experiment.

Environment Setup

The conda environment can be created from the provided environment.yml file.

Create the environment:

conda env create -f environment.yml

Activate the environment:

conda activate <env_name>

The environment name can be found in the first line of environment.yml, for example:

name: your_env_name

If you want to specify a new environment name manually, you can use:

conda env create -f environment.yml -n ec_proj
conda activate ec_proj
Data Preparation

The required data directory should be placed under:

./data/GLIF_V1_network

The expected project structure is:

EC_proj/
├── run_vcd_4rec.py
├── environment.yml
├── brainpy_impl/
├── common/
├── tensorflow_impl/
└── data/
    └── GLIF_V1_network/

Note: The dataset is usually large and is not included in this repository. Please manually place the required V1 / GLIF network data under:

./data/GLIF_V1_network
Required Path Modifications

Before running the project, make sure the data paths are correctly set.

1. Modify brainpy_impl/load_sparse.py

In:

brainpy_impl/load_sparse.py

Make sure both h5path and path point to:

"./data/GLIF_V1_network"

For example:

h5path = "./data/GLIF_V1_network"
path = "./data/GLIF_V1_network"
2. Modify the data path in run_vcd_4rec.py

In:

run_vcd_4rec.py

Make sure the data directory is also set to:

"./data/GLIF_V1_network"

For example, if the script contains a default data path such as:

DEFAULT_DATA_DIR = ...

or a config field such as:

"data_dir": ...

change it to:

"./data/GLIF_V1_network"
Running the Main Script

After setting up the environment and preparing the data, run:

python run_vcd_4rec.py

If the script supports command-line arguments, you can further customize the run according to the configuration options inside run_vcd_4rec.py.

Notes
run_vcd_4rec.py is the main script for running the current experiment.
environment.yml is used to recreate the conda environment.
The data path should consistently point to:
./data/GLIF_V1_network
Large files such as datasets, cache files, logs, and output folders should not be committed to GitHub.