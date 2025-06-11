#!/bin/bash
# FedPylot by Cyprien Quéméneur, GPL-3.0 license
# Example usage: sbatch run_centralized.sh

#SBATCH --nodes=1                        # total number of nodes (only 1 in the centralized setting)
#SBATCH --gpus-per-node=v100l:1          # total of 1 GPU
#SBATCH --ntasks-per-gpu=1               # 1 process is launched
#SBATCH --cpus-per-task=8                # CPU cores per process
#SBATCH --mem-per-cpu=2G                 # host memory per CPU core
#SBATCH --time=3-12:00:00                # time (DD-HH:MM:SS)
#SBATCH --mail-user=myemail@gmail.com    # receive mail notifications
#SBATCH --mail-type=ALL

# Check GPU on compute node
nvidia-smi

# Load modules
module load mpi/openmpi-x86_64

# Prepare directory to backup results
saving_path=$(pwd)/results/kitti/yolov7/centralized
mkdir -p $saving_path



# Download pre-trained weights
if [[ $SLURM_PROCID -eq 0 ]]; then
    bash weights/get_weights.sh yolov7
fi

#    --weights weights/yolov7/yolov7_training.pt \

# Run centralized experiment (see yolov7/train.py for more details on the settings)
python yolov7/train.py \
    --client-rank 0 \
    --epochs 150 \
    --data data/kitti.yaml \
    --batch 256 \
    --weights weights/yolov7/yolov7-tiny.pt \
    --img 640 640 \
    --cfg yolov7/cfg/training/yolov7-tiny.yaml \
    --hyp data/hyps/hyp.scratch.clientopt.kitti.yaml \
    --workers 16 \
    --multi-scale \
    --project experiments \
    --name 'Train Server' \
    --device 0,1,2,3,4,5,6,7 \
    --cache-images \
    --rect \
    --save_period 10 \


# Backup experiment results to network storage
cp -r ./experiments $saving_path
