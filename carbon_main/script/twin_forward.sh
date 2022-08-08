python pctm_v00.py \
--year 2019 \
--month 7 \
--day 1 \
--last_day 7 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "True" \
--sim_dimension 3 \
--layer_type "layer_47" \
--if_plot_result_str "True" \
--plot_interval 1 \
--data_folder '/Users/yaoyichen/dataset/auto_experiment/experiment_0/' \
--sub_data_folder "201907" \
--result_folder '/Users/yaoyichen/Desktop/auto_experiment/experiment_0/twin_inversion_forward_v04/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--flux_type "init_constant" \
--cinit_type "geos-chem" \
--problem_type "forward_simulation" \
--experiment_type "twin" \
--obs_type "full" \
--iteration_number 50 \
--early_stop_value 0.1e-10 \
--lr_flux 1e-3 \
--if_background "False" \
--background_weight 1.0