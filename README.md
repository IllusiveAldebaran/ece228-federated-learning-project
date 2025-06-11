**Disclaimer**: This project was run on a proper server with multiple GPU. We do not recommend running this on your laptop or another system as we did not test it on that hardware. This code was run on the same server, *Intel(R) Xeon(R) Silver 4110* with 8x*2080 Ti*s. 

Project implemented as HPC program. Our code continues to use the MPI communication based on this code, but restricts the testing to one GPU. Technically, if we had multiple servers, we could run this same code targetting different systems and making use of a single GPU on each system to. But for this project we had our programs split up to monopolize a single GPU.

Program uses YOLOv7-tiny under the hood. Handles the GPU allocation, model splitting, hyperparameter tuning, initialization, dataset preprocessing, and the communication between parallelized model training. 

### File Structure

`yolov7/`: Contains the YOLOv7 model with a tweaked code for exporting weights, biases, training accuracy, and a lot of parameters useful for measuring the model.
`run_federated.sh`: Script used to run the federated model. It is preferred to switch over to the corresponding branch before running, just as some of the federated model points a slightly different hyperparameters.
`run_centralized.sh`: Script used to run the centralized model. It is preferred to switch over to the corresponding branch before running for the same reason as before.
`datasets/`: Contains the kitti dataset. This project did not use the nuimages dataset, but it should be tested as well.
`data/`: Contains class information about kitti dataset. Inside `hyps` are the hyperparameters for the model. These can be tweaked for better performance.


### Instructions

**Dependencies**

An implementation of [MPI](https://carleton.ca/rcs/rcdc/introduction-to-mpi/). Most servers can install `openmpi`. 

The rest can be installed with python or should already come preinstalled on a server. You may need a few NVIDIA drivers.

You must first clone the repo. We use a python virtual environment and not conda. While we tried mixing the two, this led to a lot of errors. In the end, this was our best usage.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Preparring Dataset**

We take a subset of the KITTI Vision Benchmark Suite. We've used the first, the left color images, from the [KITTI website](https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d). Place it in the `datasets/` directory. Run the preparation script in 

```bash
python datasets/prepare_kitti.py --tar
```
This script can be editted to split up the number of clients for federated learning. By defualt, it is 5.

To launch a job the bash scripts can be started:
`./run_federated.sh`


## Acknowledgements

A lot of this project is based off the [fedpylot paper](https://arxiv.org/abs/2406.03611) work. And without [YOLOv7](https://github.com/WongKinYiu/yolov7) this code would be unable to run at all.

## License

Legally obligated to be licensed under the [GPL-3.0 License](LICENSE).
