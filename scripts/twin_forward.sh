python main.py \
--year 2019 \
--month 7 \
--day 1 \
--last_day 7 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "True" \
--sim_dimension 2 \
--layer_type "layer_1" \
--if_plot_result_str "True" \
--plot_interval 1 \
--data_folder './data/' \
--sub_data_folder "201907" \
--result_folder './result/twin_forward/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--flux_type "carbon_tracker" \
--flux_file_name "monthly_flux_np/CT2019B.flux1x1.201807_reshape.npy" \
--cinit_type "geos-chem" \
--problem_type "forward_simulation" \
--experiment_type "twin" \
--obs_type "full" 