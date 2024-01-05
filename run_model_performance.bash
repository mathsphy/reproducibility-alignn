#!/bin/bash

for target in formation_energy_peratom optb88vdw_bandgap optb88vdw_total_energy ehull mbj_bandgap bulk_modulus_kv shear_modulus_gv magmom_oszicar slme spillage kpoint_length_unit encut epsx epsy epsz mepsx mepsy mepsz dfpt_piezo_max_dielectric dfpt_piezo_max_dij dfpt_piezo_max_eij exfoliation_energy max_efg avg_elec_mass avg_hole_mass n-Seebeck n-powerfact p-Seebeck p-powerfact;do 
	for random_state in {0..4};do

output_dir="output/$target/$random_state"

mkdir -p $output_dir

if [ -f $output_dir/prediction_results_test_set.csv ];then
	echo "Found $output_dir/prediction_results_test_set.csv, skipping"
	continue
else
	echo "working on $output_dir/prediction_results_test_set.csv"
	#continue
fi


python reproduce_jdft2021.py --target $target --random_state $random_state

wd="$PWD/$target"

# Set up the config file.
config=$output_dir/config.json
cat > $config << EOF
{
    "version": "112bbedebdaecf59fb18e11c929080fb2f358246",
    "dataset": "user_data",
    "target": "target",
    "atom_features": "cgcnn",
    "neighbor_strategy": "k-nearest",
    "id_tag": "jid",
    "random_seed": $random_state,
    "classification_threshold": null,
    "n_val": null,
    "n_test": null,
    "n_train": null,
    "train_ratio": 0.8,
    "val_ratio": 0.1,
    "test_ratio": 0.1,
    "target_multiplication_factor": null,
    "epochs": 200,
    "batch_size": 64,
    "weight_decay": 1e-05,
    "learning_rate": 0.001,
    "filename": "sample",
    "warmup_steps": 2000,
    "criterion": "mse",
    "optimizer": "adamw",
    "scheduler": "onecycle",
    "pin_memory": false,
    "save_dataloader": false,
    "write_checkpoint": true,
    "write_predictions": true,
    "store_outputs": true,
    "progress": true,
    "log_tensorboard": false,
    "standard_scalar_and_pca": false,
    "use_canonize": true,
    "num_workers": 8,
    "cutoff": 8.0,
    "max_neighbors": 12,
    "keep_data_order": false,
    "model": {
        "name": "alignn",
        "alignn_layers": 4,
        "gcn_layers": 4,
        "atom_input_features": 92,
        "edge_input_features": 80,
        "triplet_input_features": 40,
        "embedding_features": 64,
        "hidden_features": 256,
        "output_features": 1,
        "link": "identity",
        "zero_inflated": false,
        "classification": false
    }
}
EOF
train_folder.py --root_dir "$wd" --config "$config" --output_dir "$output_dir" > $output_dir/output 2>&1 

rm -r $target

done
done

