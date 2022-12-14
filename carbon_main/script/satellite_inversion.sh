python pctm_v00.py \
--year 2018 \
--month 1 \
--day 1 \
--last_day 3 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "False" \
--sim_dimension 2 \
--layer_type "layer_1" \
--if_plot_result_str "True" \
--plot_interval 24 \
--data_folder '/Users/yaoyichen/dataset/auto_experiment/experiment_0/' \
--sub_data_folder '201801' \
--result_folder '/Users/yaoyichen/Desktop/auto_experiment/experiment_0/real_inversion_satellite_20180101/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--cinit_type "from_file" \
--cinit_file "/Users/yaoyichen/Desktop/auto_experiment/experiment_0/real_assimilation_satellite_20180101/assimilation_result_ocoobs_3e-5_cinit049.pt" \
--flux_type "carbon_tracker" \
--flux_file_name "monthly_flux_np/CT2019B.flux1x1.201801_reshape.npy" \
--init_flux_file_name "monthly_flux_np/CT2019B.flux1x1.201701_reshape.npy" \
--problem_type "inversion" \
--experiment_type "real" \
--obs_type "satellite" \
--flux_type_inversion "init_constant" \
--iteration_number 200 \
--early_stop_value 0.3e-6 \
--lr_flux 0.003e-4 \
--if_background "True" \
--background_weight 0.3 \
> ./logs/CliMartV00_ATT_short.log 2>&1 &

