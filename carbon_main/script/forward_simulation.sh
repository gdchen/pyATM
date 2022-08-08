python pctm_v00.py \
--year 2019 \
--month 7 \
--day 1 \
--last_day 15 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "False" \
--sim_dimension 2 \
--layer_type "layer_1" \
--if_plot_result_str "True" \
--plot_interval 24 \
--data_folder '/Users/yaoyichen/dataset/auto_experiment/experiment_0/' \
--result_folder '/Users/yaoyichen/Desktop/auto_experiment/experiment_0/forward_test_20day_0flux/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--flux_type "init_constant" \
--cinit_type "geos-chem" \
--problem_type "forward_simulation"