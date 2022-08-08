import os
 

def generate_command_str(monthstr,daystr ):
    command_str = """
    monthstr="{monthstr}"
    daystr="{daystr}"
    python pctm_v00.py \
    --year 2018 \
    --month {month_int} \
    --day {day_int} \
    --last_day 3 \
    --interval_minutes 30 \
    --device "cpu" \
    --if_mixing_str "False" \
    --sim_dimension 2 \
    --layer_type "layer_1" \
    --if_plot_result_str "True" \
    --plot_interval 24 \
    --data_folder '/home/eason.yyc/data/auto_experiment/experiment_0/' \
    --sub_data_folder '2018{monthstr}' \
    --result_folder '/home/eason.yyc/data/auto_experiment/experiment_0/full2018/real_assimilation_satellite_2018{monthstr}{daystr}/' \
    --geoschem_co2_file "GEOSChem.SpeciesConc.20190701_0000z.nc4" \
    --flux_type "carbon_tracker" \
    --flux_file_name 'monthly_flux_np/CT2019B.flux1x1.2018{monthstr}_reshape.npy' \
    --problem_type "assimilation" \
    --experiment_type "real" \
    --obs_type "satellite" \
    --cinit_type_assimilation "constant_obs" \
    --iteration_number 50 \
    --early_stop_value 0.2e-6 \
    --lr_cinit 2e-5 \
    --if_background "True" \
    --background_weight 0.5 \
    > ./logs/real_assimilation_satellite_2018{monthstr}{daystr}.log 2>&1 &
    """.format(monthstr = monthstr,
    daystr = daystr,
    month_int = int(monthstr),
    day_int = int(daystr))

    return command_str

monthstr_list = ["01","01","02","02","03","03","04","04","05","05","06","06",
                 "07","07","08","08","09","09","10","10","11","11","12","12"]

daystr_list = ["01","16","01","15","01","16","01","16","01","16","01","16",
                "01","16","01","16","01","16","01","16","01","16","01","16"]

# monthstr = "01"
# daystr = "16"

for (monthstr, daystr) in zip(monthstr_list[0:6], daystr_list[0:6]):
    command_str = generate_command_str(monthstr, daystr)
    print(command_str)
    f=os.popen(command_str)    # 创建一个文件
    print(f.read())              # 无返回值