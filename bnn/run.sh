#!/bin/bash

#SBATCH --job-name=mfvi
#SBATCH --mem-per-cpu=64000M
#SBATCH --time=0-6:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1

#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bayesianize

lambs=(0.1)
prior_scale=1.0
aug=0

annealing_epochs_1=0
annealing_epochs_2=20

for lamb in ${lambs[*]}; do

    python scripts/cifar_resnet.py --project_name="mfvi_cifar10_resnet18" \
                                --obj="elbo" \
                                --lamb=$lamb \
                                --prior_scale=$prior_scale \
                                --aug=$aug \
                                --num_epochs=100 \
                                --annealing_epochs=$annealing_epochs_1 \
                                --lr=0.005 \
                                --batch_size=256 \
                                --resnet=18 \
                                --cifar=10 &

    python scripts/cifar_resnet.py --project_name="mfvi_cifar10_resnet18" \
                                --obj="elbo" \
                                --lamb=$lamb \
                                --prior_scale=$prior_scale \
                                --aug=$aug \
                                --num_epochs=100 \
                                --annealing_epochs=$annealing_epochs_2 \
                                --lr=0.005 \
                                --batch_size=256 \
                                --resnet=18 \
                                --cifar=10 &

    wait

done