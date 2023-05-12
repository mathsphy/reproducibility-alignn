#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code retrieves and imports the data, and create a folder for each property with the associated id_prop.csv file and the POSCAR file for each entry.
"""

#%%
import numpy as np
import pandas as pd
from jarvis.db.figshare import data, get_db_info
from jarvis.core.atoms import Atoms
import matplotlib.pyplot as plt
import os
import argparse


parser = argparse.ArgumentParser(
    description='Reproduction of model performance for alignn',    
    )

parser.add_argument('--target', type=str, required=True)
parser.add_argument('--dataset', type=str, required=False,default='dft_3d_2021')
parser.add_argument('--random_state', type=int, required=False,default=0)
args = parser.parse_args()

prop = args.target
db_name = args.dataset
random_state=args.random_state


#%%
'''
Retrieve the data from the database.

List of properties in the database (for reference)
list_props = [
    'formation_energy_peratom', 
    'optb88vdw_bandgap', 
    'optb88vdw_total_energy',
    'ehull',
    'mbj_bandgap', 
    'bulk_modulus_kv', 
    'shear_modulus_gv',
    'magmom_oszicar',
    'slme', 
    'spillage', 
    'kpoint_length_unit',
    'encut',
    'epsx', 'epsy', 'epsz',
    'mepsx', 'mepsy', 'mepsz', 
    # (DFPT:elec+ionic)
    'dfpt_piezo_max_dij',
    'dfpt_piezo_max_eij',
    'exfoliation_energy',
    'max_efg',
    'avg_elec_mass', 
    'avg_hole_mass',
    'n-Seebeck', 
    'n-powerfact', 
    'p-Seebeck', 
    'p-powerfact',
]
'''

db_name='dft_3d_2021'


# check if file exists
if not os.path.exists(f'{db_name}.pkl'):
    avail_db = get_db_info().keys()
    # pretty print
    print('Available databases:')
    for db in avail_db:
        print(f'    {db}')

    d = data(db_name)
    df = pd.DataFrame(d)
    df['atoms'] = df['atoms'].apply(lambda x: Atoms.from_dict(x))
    # replace all 'na' with np.nan
    df = df.replace('na', np.nan)
    df.to_pickle(f'{db_name}.pkl')
else:
    df = pd.read_pickle(f'{db_name}.pkl')   

prop_df = df.set_index('jid')[['atoms',prop]].dropna()

#%%


#%%

# Create a directory to store the results
if not os.path.exists(prop):
    os.mkdir(prop)

# Create a id_prop.csv file under the directory, without header
prop_df[prop].to_csv(f'{prop}/id_prop.csv', header=False)

# loop over all entries of prop_df
for jid, row in prop_df.iterrows():
    atoms = row['atoms']
    atoms.write_poscar(f'{prop}/{jid}')
