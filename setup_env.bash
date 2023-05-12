#!/bin/bash

env_name="alignn_repro"
conda create --name $env_name python=3.10 -y
conda activate $env_name
pip install dgl -f https://data.dgl.ai/wheels/cu116/repo.html
pip install dglgo -f https://data.dgl.ai/wheels-test/repo.html
pip install alignn 

