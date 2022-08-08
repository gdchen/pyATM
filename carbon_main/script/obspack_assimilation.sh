python pctm_v00.py \
--year 2018 \
--month 1 \
--day 1 \
--last_day 1 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "False" \
--sim_dimension 2 \
--layer_type "layer_1" \
--if_plot_result_str "True" \
--plot_interval 24 \
--data_folder '/Users/yaoyichen/dataset/auto_experiment/experiment_0/' \
--result_folder '/Users/yaoyichen/Desktop/auto_experiment/experiment_0/obspack_layer1_assimilation/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--flux_type "init_constant" \
--problem_type "assimilation" \
--experiment_type "real" \
--obs_type "obspack" \
--cinit_type_assimilation "constant_obs" \
--iteration_number 20 \
--early_stop_value 0.1e-10 \
--lr_cinit 1e-5 \
--if_background "True" \
--background_weight 5.0