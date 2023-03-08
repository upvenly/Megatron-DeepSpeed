#!/bin/bash

#SBATCH -p ty_zhiyuan
#SBATCH -N 88
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=32
#SBATCH --mem 0
#SBATCH --gres=dcu:4
#SBATCH -J gpt3
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
ulimit -u 200000
#ethtool -g eno1

#export MIOPEN_USER_DB_PATH=/tmp/miopen-udb
#export MIOPEN_CUSTOM_CACHE_DIR=/tmp/miopen-cache
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=ib0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export OMP_NUM_THREADS=1
echo %j
echo "START TIME: $(date)"
hostfile=./hostfile_dir/$SLURM_JOB_ID
scontrol show hostnames $SLURM_JOB_NODELIST > ${hostfile}
rm `pwd`/hostfile_dir/hostfile-dl-* -f

for i in `cat $hostfile`
do
    echo ${i} slots=4 >> `pwd`/hostfile_dir/hostfile-dl-$SLURM_JOB_ID
done
np=$(cat $hostfile|sort|uniq |wc -l)

np=$(($np*4))

nodename=$(cat $hostfile |sed -n "1p")
echo $nodename
dist_url=`echo $nodename | awk '{print $1}'`

which mpirun
echo "dist_url: $dist_url"
echo "np: $np"
mpirun -np $np --allow-run-as-root --hostfile `pwd`/hostfile_dir/hostfile-dl-$SLURM_JOB_ID --bind-to none `pwd`/single.sh $dist_url
echo "END TIME: $(date)"

#just for rename log
now=$(date +"%Y%m%d_%H%M%S")
cp logs/$SLURM_JOB_ID.out logs/gpt3-$now-$SLURM_JOB_ID.log
cp logs/$SLURM_JOB_ID.err logs/gpt3-$now-$SLURM_JOB_ID.err