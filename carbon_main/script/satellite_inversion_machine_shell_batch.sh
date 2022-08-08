echo "开始测试......1"
monthstr="02"
daystr="20"
python pctm_v00.py \
--year 2018 \
--month $((10#$monthstr)) \
--day $((daystr)) \
--last_day 9 \
--interval_minutes 30 \
--device "cpu" \
--if_mixing_str "False" \
--sim_dimension 2 \
--layer_type "layer_1" \
--if_plot_result_str "True" \
--plot_interval 24 \
--data_folder '/home/eason.yyc/data/auto_experiment/experiment_0/' \
--sub_data_folder '2018'$monthstr \
--result_folder '/home/eason.yyc/data/auto_experiment/experiment_0/full2018/real_inversion_satellite_2018'$monthstr$daystr'/' \
--geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
--cinit_type "from_file" \
--cinit_file "/home/eason.yyc/data/auto_experiment/experiment_0/full2018/real_assimilation_satellite_2018"$monthstr$daystr"/assimilation_result_ocoobs_3e-5_cinit049.pt" \
--flux_type "carbon_tracker" \
--flux_file_name "monthly_flux_np/CT2019B.flux1x1.2018"$monthstr"_reshape.npy" \
--init_flux_file_name "monthly_flux_np/CT2019B.flux1x1.2017"$monthstr"_reshape.npy" \
--problem_type "inversion" \
--experiment_type "real" \
--obs_type "satellite" \
--flux_type_inversion "init_constant" \
--iteration_number 80 \
--early_stop_value 0.2e-6 \
--lr_flux 0.003e-4 \
--if_background "True" \
--background_weight 0.3 \
> ./logs/real_inversion_satellite_2018$monthstr$daystr.log 2>&1 &
wait
echo "结束测试"