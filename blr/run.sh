#!/bin/bash

#SBATCH --job-name=mfvi_mlp
#SBATCH --mem-per-cpu=64000M
#SBATCH --time=0-3:00:00
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cpe_vi_tf

lambs=(0.1 1.0 10.0 100.0 1000.0)
seed1=1
prior_scale1=1.0

for lamb in ${lambs[*]}; do

    python mlp.py --project_name="mfvi_mlp" \
                  --seed=$seed1 \
                  --dataset="fmnist" \
                  --algorithm=1 \
                  --num_epochs=100 \
                  --num_hidden_units=20 \
                  --lamb=$lamb \
                  --prior_scale=$prior_scale1 &

    python mlp.py --project_name="mfvi_mlp" \
                  --seed=$seed1 \
                  --dataset="fmnist" \
                  --algorithm=2 \
                  --num_epochs=100 \
                  --num_hidden_units=20 \
                  --lamb=$lamb \
                  --prior_scale=$prior_scale1 &

    python mlp.py --project_name="mfvi_mlp" \
                  --seed=$seed1 \
                  --dataset="fmnist" \
                  --algorithm=3 \
                  --num_epochs=100 \
                  --num_hidden_units=20 \
                  --lamb=$lamb \
                  --prior_scale=$prior_scale1 &

    wait

done