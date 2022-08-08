monthstr="01"
daystr="16"
python pctm_v00.py \
--year 2018 \
--month $((monthstr)) \
--day $((daystr)) \
--last_day 3 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "False" \
--sim_dimension 2 \
--layer_type "layer_1" \
--if_plot_result_str "True" \
--plot_interval 24 \
--data_folder '/home/eason.yyc/data/auto_experiment/experiment_0/' \
--sub_data_folder '2018'$monthstr \
--result_folder '/home/eason.yyc/data/auto_experiment/experiment_0/full2018/real_assimilation_satellite_2018'$monthstr$daystr'/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--flux_type "carbon_tracker" \
--flux_file_name 'monthly_flux_np/CT2019B.flux1x1.2018'$monthstr'_reshape.npy' \
--problem_type "assimilation" \
--experiment_type "real" \
--obs_type "satellite" \
--cinit_type_assimilation "constant_obs" \
--iteration_number 50 \
--early_stop_value 0.2e-6 \
--lr_cinit 2e-5 \
--if_background "True" \
--background_weight 0.5 \
> ./logs/real_assimilation_satellite_2018$monthstr$daystr.log 2>&1 &

