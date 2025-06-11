#!/bin/bash
# FedPylot by Cyprien Quéméneur, GPL-3.0 license
# Example usage: sbatch run_centralized.sh


# Run centralized experiment (see yolov7/train.py for more details on the settings)
python yolov7/detect.py \
    --weights experiments/weights/best.pt \
    --source datasets/kitti/server/images \
    --img-size 640 \
    --conf-thres 0.25 \
    --save-txt \
    --save-conf \
    --project experiments/results \
    --name detect_results

