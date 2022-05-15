for n in 5 6
do
    for se in 0 1 2
    do
        let "tar=${se}%$n"
        CUDA_VISIBLE_DEVICES=$1 python3 src/main.py --config="maven_sep" --env-config=tiny_climb with env_args.max_landmarks=$n env_args.num_landmarks=$n seed=${se} env_args.target_id=${tar} env_args.map_name="2A${n}L_MAVEN"
    done
done
