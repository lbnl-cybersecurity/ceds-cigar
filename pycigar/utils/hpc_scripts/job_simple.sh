#!/bin/bash

#SBATCH --job-name=pycigar_unbalance
#SBATCH --account=pc_cigar
#SBATCH --mail-user=amilesi@lbl.gov
#SBATCH --qos=lr_normal
#SBATCH --partition=lr6
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks-per-node 1

conda init bash && source ~/.bashrc
# uncomment to git pull before each job
# eval "$(ssh-agent -s)" 
# ssh-add ~/.ssh/id_rsa &
# cd ceds-cigar && git pull

conda activate pycigar
#python ~/ceds-cigar/pycigar/notebooks/unbalance/unbalance_multiagent.py --workers 8 --save-path ~/results_multiagent_eval_new_r --eval-interval 20 --epochs 1000
python ~/ceds-cigar/pycigar/notebooks/unbalance/unbalance.py --continuous --workers 8 --save-path ~/results_continuous_eval_30 --eval-interval 40 --epochs 1000
#python ~/ceds-cigar/pycigar/notebooks/unbalance/unbalance.py --workers 8 --save-path ~/results_discrete_eval_30 --eval-interval 40 --epochs 1000

