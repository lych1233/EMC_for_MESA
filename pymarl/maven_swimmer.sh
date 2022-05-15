algo="maven_sep"
for se in 0 1 2
do
    CUDA_VISIBLE_DEVICES=$1 python3 src/main.py --config=${algo} --env-config=ma_swimmer with seed=${se} env_args.map_name="maven_swimmer_naive"
done