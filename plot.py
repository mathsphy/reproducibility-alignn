

#%%
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
root_dir = '~/Gdrive/UofT/Coding/auto_alloys/paper-reproducibility-alignn/output'
os.chdir(os.path.expanduser(root_dir))
list_props = 'formation_energy_peratom optb88vdw_bandgap optb88vdw_total_energy ehull mbj_bandgap bulk_modulus_kv shear_modulus_gv magmom_oszicar slme spillage kpoint_length_unit encut epsx epsy epsz mepsx mepsy mepsz dfpt_piezo_max_dielectric dfpt_piezo_max_dij dfpt_piezo_max_eij exfoliation_energy max_efg avg_elec_mass avg_hole_mass n-Seebeck n-powerfact p-Seebeck p-powerfact'.split()
#%%

db_name='dft_3d_2021'
df = pd.read_pickle(f'../{db_name}.pkl') 


#%%
# # For each column, get the number of entries without NaN
counts = {col:df[col].dropna().count() for col in list_props}
counts = pd.Series(counts).sort_values()
# # plot counts using barplot
# %matplotlib inline

# count the MAD values
# MAD = {col:df[col].dropna().mad() for col in list_props}

# counts.plot.bar()
# plt.ylabel('Number of entries')
# plt.title('Number of entries for each property')
# plt.show()


#%%
'''
read in the MAE scores
'''
root_dir = '~/Gdrive/UofT/Coding/auto_alloys/paper-reproducibility-alignn/output'
root_dir = os.path.expanduser(root_dir)
os.chdir(os.path.expanduser(root_dir))

# formation_energy_peratom optb88vdw_bandgap optb88vdw_total_energy ehull mbj_bandgap bulk_modulus_kv shear_modulus_gv magmom_oszicar slme spillage kpoint_length_unit encut epsx epsy epsz mepsx mepsy mepsz dfpt_piezo_max_dielectric dfpt_piezo_max_dij dfpt_piezo_max_eij exfoliation_energy max_efg avg_elec_mass avg_hole_mass n-Seebeck n-powerfact p-Seebeck p-powerfact
props = {
    'formation_energy_peratom': ['Formation energy',0.033],
    'optb88vdw_bandgap': ['Bandgap (OPT)',0.140],
    'optb88vdw_total_energy': ['Total energy (OPT)',0.037],
    'ehull': ['Ehull',0.076],
    'mbj_bandgap': ['Bandgap (MBJ)',0.31],
    'bulk_modulus_kv': [r'$K_v$',10.40],
    'shear_modulus_gv': [r'$G_v$',9.48],
    'magmom_oszicar': ['Magmom',0.26],
    'slme': ['SLME',4.52],
    'spillage': ['Spillage',0.35],
    'kpoint_length_unit': ['Kpoint length',9.51],
    'encut': ['Plane-wave cutoff',133.8],
    'epsx': [r'$\epsilon_x$ (OPT)',20.40],
    'epsy': [r'$\epsilon_y$ (OPT)',19.99],
    'epsz': [r'$\epsilon_z$ (OPT)',19.57],
    'mepsx': [r'$\epsilon_x$ (MBJ)',24.05],
    'mepsy': [r'$\epsilon_y$ (MBJ)',23.65],
    'mepsz': [r'$\epsilon_z$ (MBJ)',23.73],
    'dfpt_piezo_max_dielectric': [r'$\epsilon$ (elec+ionic)',28.15],
    'dfpt_piezo_max_dij': [r'$d_{ij}$',20.57],
    'dfpt_piezo_max_eij': [r'$e_{ij}$',0.147],
    'exfoliation_energy': ['Exf. energy',51.42],
    'max_efg': ['Max. EFG',19.12],
    'avg_elec_mass': [r'Avg. $m_e$',0.085],
    'avg_hole_mass': [r'Avg. $m_h$',0.124],
    'n-Seebeck': ['n-SB',40.92],
    'n-powerfact': ['n-PF',442.30],
    'p-Seebeck': ['p-SB',42.42],
    'p-powerfact': ['p-PF',440.26]
}

# Convert to df
df = pd.DataFrame.from_dict(props, orient='index')
df.columns = ['name', 'MAE']
df['keys'] = df.index 
df['counts'] = counts
df = df.sort_values(by='counts', ascending=True)

#%%

for random_state in range(5):
    mae = []
    for prop in df.index:
        # read in the MAE from the file 'root_dir/prop/random_state/output'
        file0 = f'{prop}/{random_state}/output'
        file1 = f'{prop}/{random_state}/prediction_results_test_set.csv'
        # if file exists
        if os.path.exists(file1):

            # Use bash command to get the MAE
            mae0 = os.popen(f'grep "Test MAE" {file0} | awk \'{{print $NF}}\'').read()
                    
            if mae0 == '':                
                if os.path.exists(file1):
                    # read file and get the MAE
                    df0 = pd.read_csv(file1)
                    mae0 = (df0['target'] - df0['prediction']).abs().mean()

            mae.append(float(mae0))
        else:
            print(f'No file found in {root_dir}/{file1}')
            mae.append(np.nan)

    df[f'MAE_{random_state}'] = mae
    df[f'ratio_{random_state}'] = df[f'MAE_{random_state}'] / df['MAE']


df.to_csv('MAE.csv')

#%%


df['ratio_mean'] = df[[f'ratio_{i}' for i in range(5)]].mean(axis=1) -1 
df['ratio_std'] = df[[f'ratio_{i}' for i in range(5)]].std(axis=1)
df['ratio_min'] = df[[f'ratio_{i}' for i in range(5)]].min(axis=1) -1 
df['ratio_max'] = df[[f'ratio_{i}' for i in range(5)]].max(axis=1) -1 
df['ratio_abs_min'] = (df[[f'ratio_{i}' for i in range(5)]]-1).abs().min(axis=1) 
df['ratio_mean_abs'] = df['ratio_mean'].abs()
# plot the MAE's statistics
fig, ax = plt.subplots(figsize=(4.*2, 5))
# scatter plot of the min and max
ax.scatter(df['name'], df['ratio_max'], c='k', marker='v', label='max')
ax.scatter(df['name'], df['ratio_min'], c='k', marker='^', label='min')
# error bar of the mean and std
ax.errorbar(df['name'], df['ratio_mean'], yerr=df['ratio_std'], fmt='s', c='k', label='mean & std')
# set yticks
ax.set_yticks(np.arange(-0.3, 0.4, 0.1))
# transform the yticks to percentage
vals = ax.get_yticks()
ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
# also show minor yticks
ax.set_yticks(np.arange(-0.3, 0.4, 0.05), minor=True)
ax.set_ylim(-0.3, 0.3)

# y axis label
ax.set_ylabel('Deviation in MAE (%)')
# add a second y axis and plot counts.loc[df.index]
ax2 = ax.twinx()
ax2.bar(df['name'], counts.loc[df.index], alpha=0.35, color='b', label='Number of entries')
# change the 2nd y axis into scientific notation with decimal
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0), useMathText=True)
# y axis label
ax2.set_ylabel('Number of entries', color='b')
# set ylim
ax2.set_ylim(0, 60000)
# set xlim
ax.set_xlim(-1, 29) 
# change the color of the 2nd y axis ticks and labels
ax2.tick_params(axis='y', colors='b')

# rotate the xticks
ax.set_xticklabels(df['name'], rotation=90, ha='center')
# legend
ax.legend(loc=(0.15,0.71))
# add grid
ax.grid(linewidth=0.5,which='both')
fig.savefig('../MAE.pdf', bbox_inches='tight', dpi=200)
#%%
x1 = (df['ratio_abs_min'].sort_values(ascending=True) * 100).tolist()
# x2 = (df['ratio_mean_abs'].sort_values(ascending=True) * 100).tolist()
y = [i+1 for i in range(len(x1))]
fig, ax = plt.subplots(figsize=(4, 5))
ax.plot(x1,y,'o-',label='min abs deviation')
# ax.plot(x2,y,'^-',label='mean abs deviation')
ax.set_ylim(0,30)
ax.set_xlim(0,6)





#%%
ratio_mean = df['ratio_mean'].abs().sort_values() * 100
ratio_mean



#%%

layer_o = pd.read_csv('/home/kangming/Gdrive/UofT/Coding/auto_alloys/paper-reproducibility-alignn/output/layer_original.csv', index_col=0)
layer_o.columns=[0,1,2,3,4]
layer_o = layer_o / layer_o.loc[4,4]

layert_o = pd.read_csv('/home/kangming/Gdrive/UofT/Coding/auto_alloys/paper-reproducibility-alignn/output/time_original.csv', index_col=0)
layert_o.columns=[0,1,2,3,4]
layert_o = layert_o / layert_o.loc[4,4]

#%%


#%%
'''
ablation study
'''
os.chdir('/home/kangming/Gdrive/UofT/Coding/auto_alloys/paper-reproducibility-alignn/output/ablation')

# construct a df from the data, index is the number of alignn layers, columns are the number of gcn layers
df = pd.DataFrame(index=range(5), columns=range(5))
df_time = pd.DataFrame(index=range(5), columns=range(5))
for alignn_layers in range(5):
    for gcn_layers in range(5):
        file = f'{alignn_layers}.{gcn_layers}/output'
        mae0 = float(os.popen(f'grep "Test MAE" {file} | awk \'{{print $NF}}\'').read())
        df.loc[alignn_layers, gcn_layers] = mae0
        
        epoch = int(os.popen(f'grep "Train_MAE" {file} | wc -l').read())
        times = float(os.popen(f'grep "Time taken" {file} | awk \'{{print $NF}}\'').read())
        time_per_epoch = times / epoch / 60
        df_time.loc[alignn_layers, gcn_layers] = time_per_epoch

df_time = df_time / df_time.loc[4,4]              
df = df / df.loc[4,4]

#%%
def get_pareto(x,y):
    # sort the points in ascending order of x
    points = sorted(zip(x, y))

    # compute the Pareto front
    pareto_front = [points[0]]
    for point in points[1:]:
        if point[1] < pareto_front[-1][1]:
            pareto_front.append(point)

    # unpack the pareto front points
    px, py = zip(*pareto_front)
    return px, py


#%%
scatter_size = 10
fontsize = 6
# plot df vs df_time
fig, ax = plt.subplots(figsize=(3.5, 3.5))
# use unfilled circle for the data points
ax.scatter(df_time.values.ravel(), df.values.ravel(), 
           facecolors='none', edgecolors='k',label='Reproduced', marker='s',s=scatter_size)
px,py = get_pareto(df_time.values.ravel(), df.values.ravel())
ax.plot(px, py, c='k', alpha=0.3)

# add the coordinates of the data points
for i in range(5):
    for j in range(5):
        if df_time.loc[i,j] < 0.45 and not (i==0 and j==0):
            ax.text(df_time.loc[i,j]-0.08, df.loc[i,j]-0.01, f'[{i},{j}]', fontsize=fontsize)
i,j = 2,0
ax.text(df_time.loc[i,j]-0.073, df.loc[i,j]-0.02, f'[{i},{j}]', fontsize=fontsize)
i,j = 1,3
ax.text(df_time.loc[i,j]-0.045, df.loc[i,j]-0.035, f'[{i},{j}]', fontsize=fontsize, color='k')
i,j = 1,4
ax.text(df_time.loc[i,j]-0.0125, df.loc[i,j]+0.015, f'[{i},{j}]', fontsize=fontsize, color='k')
i,j = 2,1
ax.text(df_time.loc[i,j]-0.01, df.loc[i,j]-0.026, f'[{i},{j}]', fontsize=fontsize, color='k', ha='center',va='center')
i,j = 2,2
ax.text(df_time.loc[i,j], df.loc[i,j]+0.022, f'[{i},{j}]', fontsize=fontsize, color='k', ha='center',va='center')


# use unfilled square for the data points
ax.scatter(layert_o.values.ravel(), layer_o.values.ravel(), facecolors='none', 
           edgecolors='r', marker='o',label='Original',s=scatter_size)
px,py = get_pareto(layert_o.values.ravel(), layer_o.values.ravel())
ax.plot(px, py, c='r', alpha=0.3)

# add the coordinates (text in red) of the data points
for i in range(5):
    for j in range(5):
        if layert_o.loc[i,j] < 0.55 and not (i==0 and j==0):
            ax.text(layert_o.loc[i,j]+0.01, layer_o.loc[i,j]-0.005, f'({i},{j})', fontsize=fontsize, color='r')

i,j = 2,0
ax.text(layert_o.loc[i,j], layer_o.loc[i,j]+0.015, f'({i},{j})', fontsize=fontsize, ha='center', color='r')
i,j = 1,3
ax.text(layert_o.loc[i,j]-0.05, layer_o.loc[i,j]-0.03, f'({i},{j})', fontsize=fontsize, color='r')
i,j = 1,4
ax.text(layert_o.loc[i,j]-0.015, layer_o.loc[i,j]+0.013, f'(1,2)&({i},{j})', fontsize=fontsize, color='r')
i,j = 2,2
ax.text(layert_o.loc[i,j]-0.02, layer_o.loc[i,j]-0.032, f'({i},{j})', fontsize=fontsize, color='r', ha='center',va='center')
i,j = 2,1
ax.text(layert_o.loc[i,j]+0.035, layer_o.loc[i,j]+0.02, f'({i},{j})', fontsize=fontsize, color='r', ha='center',va='center')

# set y limit to 0.9 to 2
ax.set_ylim(0.9, 2)
# set x limit
ax.set_xlim(0.15, 1.05)
# set y label
ax.set_ylabel('Normalized MAE')
# set x label
ax.set_xlabel('Normalized Time per Epoch')
ax.legend()
fig.savefig('../../ablation.pdf', bbox_inches='tight', dpi=200)

#%%

(df/layer_o-1)*100
#%%
(df_time/layert_o-1)*100

#%%
df.to_csv('ablation.csv')


#%%
'''
Double check the data MAD values.
'''

# jdft = pd.read_pickle('dft_3d_2021.pkl')

# # Get the MAD for each property
# mad = {}
# for prop in props.keys():
#     # round to 2 decimal places
#     mad[prop] = round(jdft[prop].mad(), 2)
#     print(f'{prop}: {mad[prop]-df.loc[prop, "MAD"]}')

# %%


