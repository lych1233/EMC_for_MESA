algo="sep_qmix"
for se in 0 1 2
do
    CUDA_VISIBLE_DEVICES=$1 python3 src/main.py --config=${algo} --env-config=ma_swimmer with seed=${se} env_args.scenario_name="swimmer_velocity" env_args.map_name="qmix_swimmer_velocity"
done