#!/bin/bash

#SBATCH --job-name=pycigar_unbalance
#SBATCH --account=pc_cigar
#SBATCH --mail-user=amilesi@lbl.gov
#SBATCH --qos=lr_normal
#SBATCH --partition=lr6
#SBATCH --cpus-per-task=32
#SBATCH --nodes=5
#SBATCH --tasks-per-node 1


conda init bash && source ~/.bashrc
conda activate pycigar
eval "$(ssh-agent -s)" 
ssh-add ~/.ssh/id_rsa &
cd ceds-cigar && git pull

worker_num=4 # Must be one less that the total number of nodes

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node1=${nodes_array[0]}

ip_prefix=$(srun --nodes=1 --ntasks=1 -w $node1 hostname --ip-address) # Making address
suffix=':6379'
ip_head=$ip_prefix$suffix
redis_password=$(uuidgen)

srun --nodes=1 --ntasks=1 -w $node1 ray start --block --head --redis-port=6379 --redis-password=$redis_password & # Starting the head
sleep 120
# Make sure the head successfully starts before any worker does, otherwise
# the worker will not be able to connect to redis. In case of longer delay,
# adjust the sleeptime above to ensure proper order.

for ((  i=1; i<=$worker_num; i++ ))
do
  node2=${nodes_array[$i]}
  srun --nodes=1 --ntasks=1 -w $node2 ray start --block --address=$ip_head --redis-password=$redis_password & # Starting the workers
  # Flag --block will keep ray process alive on each compute node.
  sleep 20
done


python ~/ceds-cigar/pycigar/notebooks/unbalance.py --redis-pwd $redis_password --workers 8 --save-path ~/results_continuous_gridearch_M --eval-interval 20 --continuous --epochs 1000

