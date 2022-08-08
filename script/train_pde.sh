# experiment=""
# experiment_code="1"
# save_dir=./log/${experiment}_$(date +'%Y_%m_%d')_${experiment_code}/

# log_file=${save_dir}/log.txt

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python surrogate_1d.py \
--method "rk4" \
--batch_time 10 \
--batch_size 20 \
--niters 2000 \
--adjoint False \
--main_folder "surrogate_1d_0825" \
--sub_folder "pde" \
--prefix "pde_" \
--surrogate_mode "pde" \
--load_model False \
--save_model True \
--source_checkpoint_file_name "test1.pth.tar" \
--target_checkpoint_file_name "pde_.pth.tar" \
--test_freq 20 \
--save_interval 100 \
--plot_interval  100

