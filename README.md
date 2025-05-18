<div align="center">
  <img src="assets/logo.png" width=550 alt="FedPylot Logo">
</div>

<h2 align="center">
    <p>Federated Learning for Real-Time Object Detection in Internet of Vehicles</p>
</h2>

<p align="center">
    <a href="https://github.com/CyprienQuemeneur/fedpylot/blob/main/LICENSE"><img alt="license" src="https://img.shields.io/badge/license-GPLv3.0-blue.svg"></a>
    <a href="https://arxiv.org/abs/2406.03611"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2406.03611-b31b1b.svg"></a>
</p>

**FedPylot** is an open-source lightweight MPI-based program designed to explore and simplify the federated training of real-time object detection models, particularly for the purpose of autonomous driving applications.
If you're a young researcher or simply an enthusiast, integrating modern object detectors into advanced federated learning frameworks can be a complex and time-consuming endeavor, which might detract from your actual project.
Hopefully, FedPylot can serve as an easy-to-use foundation and help quick-start your research! 🤗

For the full details of our experiments and findings, see our [paper](https://arxiv.org/abs/2406.03611).
For questions or inquiries about FedPylot, you may contact
[cyprien.quemeneur@protonmail.com](mailto:cyprien.quemeneur@protonmail.com).

## 📖 Features

In FedPylot, federated optimization is performed on a HPC cluster, where each federated participant (client or server) maps to a compute node.
The server is responsible for initializing the shared object detection model with weights pre-trained on MS COCO, aggregating model updates, and evaluating the joint model on a set of unseen examples at the end of each communication round.
We assume full-client participation, synchronous updates, and state persistence between rounds.

- **Object detector**: YOLOv7
- **Communication backend**: MPI
- **Server-side optimizers**: FedAvg, FedAvgM, FedAdagrad, FedAdam, FedYogi
- **Local optimizers**: SGD, YOLOv7 default
- **Datasets**: KITTI (IID), nuImages (non-IID)

In addition, model updates are reduced to half-precision and protected using hybrid encryption (AES-GCM + RSA).
More advanced compression and privacy-preservation techniques are not currently supported.
The default optimizer of YOLOv7 involves SGD with Nesterov momentum, weight decay, and a one-cycle cosine annealing policy for the learning rates.

## 🐍 Installation

We recommend installing FedPylot both locally (for data preparation and prototyping) and on your HPC cluster (for running experiments).

```bash
git clone https://github.com/CyprienQuemeneur/fedpylot.git
cd fedpylot
```

To set up your local virtual environment, run:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> Installing packages on an HPC cluster often has specific considerations.
> You should refer to your cluster's documentation for best practices on package installation and module loading.
> Ideally, the virtual environment should be installed outside the project directory on the cluster.

## ⚙️ Data Preparation

FedPylot supports two prominent autonomous driving datasets out of the box:

- The 2D object detection subset of the KITTI Vision Benchmark Suite.
- nuImages, an extension of nuScenes focused on 2D object detection.

Preparing your data involves:

1. Converting annotations to the YOLO format.
2. Splitting the samples and labels among the federated participants (server and clients).

Run data preparation scripts locally to create a directory for each federated participant, containing their respective data samples and labels.
Archiving these directories for transfer and storage on the cluster is highly recommended and can be automated by the scripts.
For secure and reliable transfer of large datasets to your cluster, consider using a tool such as [Globus](https://www.globus.org/).

#### KITTI

Obtain the *left color images* and *training labels* for the 2D object detection task from the official [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d) (account required) and unzip them in the `datasets` directory of this program.

By default, 25% of the training data is allocated to the central server, as KITTI does not feature a predefined validation set.
For the remaining data, we perform a balanced and IID split among 5 clients.
The *DontCare* attribute is ignored, leaving 8 classes.
The random seed is fixed for reproducible splits.
To perform the split and annotation conversion (and optionally archive the output), run:

```bash
python datasets/prepare_kitti.py --tar
```

If you wish to customize the splitting strategy, edit `prepare_kitti.py` accordingly.

#### nuImages

Download the nuImages samples and metadata from the official [nuScenes website](https://nuscenes.org/nuimages) (account required) and unzip them in the `datasets` directory of this program.
Sweeps (non-keyframes) are not annotated and were not included in our experiments.
nuImages is structured as a relational database, so this setup uses the [nuscenes-devkit](https://github.com/nutonomy/nuscenes-devkit) to manipulate the files.
For the devkit to work properly, you need to create a `nuimages` directory and arrange the files as follows:

```text
/datasets/nuimages
    samples    -  Sensor data for keyframes (annotated images).
    v1.0-train -  JSON tables that include all the metadata and annotations for the training set.
    v1.0-val   -  JSON tables that include all the metadata and annotations for the validation set.
```

By default, the predefined nuImages validation set is assigned to the server, while the training data is split non-IID among 10 clients based on capture location and timeframes, to simulate realistic data heterogeneity.

To create splits retaining 10 classes based on the nuScenes competition map, run:

```bash
python datasets/prepare_nuimages.py --class-map 10 --tar
```

To create splits retaining the full long-tail distribution with all 23 classes, run:

```bash
python datasets/prepare_nuimages.py --class-map 23 --tar
```

If you wish to customize the splitting strategy, edit `prepare_nuimages.py` accordingly.

## 🚀 Launching a Job

We provide Slurm job script templates for both centralized and federated settings.
Experiments are initialized using official YOLOv7 pre-trained weights.

Before training, data is copied to the local storage of the compute node(s).
For federated experiments, `scatter_data.py` (an MPI script) handles dispatching the correct local datasets to each participant.

#### Pre-trained weights

The job scripts typically download model pre-trained weights.
If your cluster's compute nodes lack internet access, download them manually beforehand. FedPylot supports all YOLOv7 variants.
For example, to download weights to initialize YOLOv7-tiny, run:

```bash
bash weights/get_weights.sh yolov7-tiny
```

#### Running an experiment

To launch a federated experiment, modify `run_federated.sh` to suit your cluster's requirements and desired experimental settings.
Then submit:

```bash
sbatch run_federated.sh
```

Similarly, for the centralized learning baseline, edit `run_centralized.sh` and submit:

```bash
sbatch run_centralized.sh
```

#### Options

The federated learning settings are as follows:

- `--nrounds`: Number of communication rounds (Default: 30).
- `--epochs`: Local training epochs per client per round (Default: 5).
- `--server-opt`: Server-side optimizer choice (Default: fedavg).
- `--server-lr`: Server-side learning rate (Default: 1.0).
- `--tau`: Server-side adaptivity for FedAdagrad, FedAdam, FedYogi (Default: 0.001).
- `--beta`: Server-side momentum for FedAvgM (Default: 0.1).

The core object detection settings are as follows:

- `--architecture`: Choice of object detector architecture (Default: yolov7).
- `--weights`: Path to pre-trained weights (e.g., `weights/yolov7/yolov7_training.pt`).
- `--data`: Path to dataset YAML file (e.g., `data/kitti.yaml`).
- `--bsz-train`: Training batch size (Default: 32).
- `--bsz-val`: Validation batch size (Default: 32).
- `--img`: Image size in pixels with letterbox resizing (Default: 640).
- `--conf`: Object confidence threshold for detection (Default: 0.001).
- `--iou`: IoU threshold for NMS (Default: 0.65).
- `--cfg`: Path to model YAML file (Default: `yolov7/cfg/training/yolov7.yaml`).
- `--hyp`: Path to hyperparameters YAML file (e.g., `data/hyps/hyp.scratch.clientsgd.yaml`).
- `--workers`: Number of data loading workers (Default: 8).

More advanced options for object detection are available in `yolov7/train.py` and `yolov7/train_aux.py`.

## 🎓 Citation

If you find FedPylot is useful in your research or applications, consider giving us a star 🌟 and citing our paper.

```bibtex
@article{quemeneur2024fedpylot,
  title={{FedPylot}: Navigating Federated Learning for Real-Time Object Detection in Internet of Vehicles}, 
  author={Quéméneur, Cyprien and Cherkaoui, Soumaya},
  journal={arXiv preprint arXiv:2406.03611},
  year={2024}
}
```

## 🤝 Acknowledgements

- We sincerely thank the authors of [YOLOv7](https://github.com/WongKinYiu/yolov7) for providing their code to the community!

- FedPylot was first released as part of my Master's thesis at the [LINCS laboratory](https://lincslab.ca/) of [Polytechnique Montréal](https://polymtl.ca/), under the supervision of Prof. [Soumaya Cherkaoui](https://scholar.google.be/citations?user=fW60_n4AAAAJ).

## 📜 License

FedPylot is released under the [GPL-3.0 License](LICENSE).
