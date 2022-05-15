for seed in 1 2 3 4 5 6 7 8 9
do
    python3 src/main.py --config=EMC_naive --env-config=multi_step_10 with seed=${seed}
done