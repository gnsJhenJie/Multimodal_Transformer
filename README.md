[![CI](https://github.com/wasn-lab/Multimodal_Transformer/actions/workflows/main.yml/badge.svg)](https://github.com/wasn-lab/Multimodal_Transformer/actions/workflows/main.yml/)

# THE MOST THANKFUL THINGS
**https://github.com/StanfordASL/Trajectron-plus-plus**.<br></br>
Without this repo, I could not complete my whole project!<br></br>
All of my project code is based on this repo.

# Directory
The Directory of our project.

    ${Trajectron root}
    ├── data
    ├── experiments
    ├── Transformer
    ├── README.md
    └── requirements.txt

# Data preparation

## For dataset please download into the direct Directory.<br></br>
The setting of nuScenes dataset is quite complicate. So we zip the *FULL* Dataset and store in nas, you can downloaded by the command below or download by official website https://www.nuscenes.org/. Then unzip the file at correspond folder.<br></br>

    ${Trajectron root}
    ├── data
        `-- ├── sets --> unzip here!

Create directory :

    mkdir -p ~/data/sets
    
Download nuScene dataset:

> ```wget --no-check-certificate -O dataset.tar.xz "https://140.115.54.247/share.cgi?ssid=29b79213862b4827b5e6bacfcd784041&fid=29b79213862b4827b5e6bacfcd784041&filename=nuscenes.tar.xz&openfolder=forcedownload&ep=="```
    
# Create Training Environment

For the following process, we first need to create a fresh environment, and there are two method to choose conda or virtualenv.

## For virtualenv
**Notice!! /usr/bin/python3.6 is default path, please modify the path according to your Environment!**

    virtualenv -p /usr/bin/python3.6 py3.6_env 
    source py3.6_env/bin/activate
    pip install -r requirements_3.6.txt
    
## For conda

    conda create --name Fusion_trans python=3.6 -y
    source activate Fusion_trans
    pip install -r requirements.txt
    
## Install devkit for dataset

For nuScenes devkit.
    
    pip install nuscenes-devkit

# Training Progress


## Data Preprocessing

    ${Trajectron root}
    `-- experiments
        `--nuScene
        
To preprocess the nuScenes data, you can execute the command below in `nuScene` directory, depending on the data version you desire.<br></br>
| Typo preprocess| Command|
|:---|:---|
| Map preprocess |```python3 nuScenes_process_data.py --data=../../data/sets/nuscenes/v1.0/ --version=v1.0-trainval --output_path=../processed_data```|
| Lane preprocess |```python3 nuScenes_process_data.py --data=../../data/sets/nuscenes/v1.0/ --version=v1.0-trainval --output_path=../processed_data --lane_process```|

## Training

    ${Trajectron root}
    ├── Transformer

To train a model with nuScene data, you can execute the command below in `Transformer` directory, depending on the model version you desire.<br></br>
**Note that the device parameter depends on your machine if machine don't have gpu use "cpu".**
| Model| Command|
|:---|:---|
| Basic |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScene/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/nuScene/models --train_epochs 20 --node_freq_mult_train --log_tag basic_transformer --augment --autoregressive --device "cuda:1" ```|
| Map_encoding_CNN |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScene/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/nuScene/models --train_epochs 20 --node_freq_mult_train --log_tag CNN_transformer --augment --autoregressive --map_cnn_encoding --device "cuda:1" ```|
| Map_encoding_ViT |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScene/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/nuScene/models --train_epochs 20 --node_freq_mult_train --log_tag ViT_transformer --augment --autoregressive --map_vit_encoding --device "cuda:1" ```|
| Lane_encoding_CNN |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/nuScene/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/nuScene/models --train_epochs 20 --node_freq_mult_train --log_tag Lane_transformer --augment --autoregressive --lane_cnn_encoding --device "cuda:1" ``` |

# Evaluate Progress 
    
    ${Trajectron root}
    ├── experiments
        `-- nuScene
        
If you want to evaluate the trained model to generate trajectories and plot them, you can use `NuScenes Qualitative.ipynb` norebook.<br></br>
To evaluate a trained model's performance on forecasting vehicles, you can execute a one of the following commands from within the `experiments/nuScenes` directory.
**1. that the testing dataset will be slightly different between map and lane we filter more data in lane.**<br></br>
**2. Lane encoding CNN can not use map data but other method could use both data to evaluate.**<br></br>

| Model| Command|
|:---|:---|
| Basic |```python evaluate.py --model models/basic_transformer --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag basic_transformer --node_type VEHICLE --prediction_horizon 12```|
| Map_encoding_CNN |```python evaluate.py --model models/CNN_transformer --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag CNN_transformer --node_type VEHICLE --prediction_horizon 12```|
| Map_encoding_ViT |```python evaluate.py --model models/ViT_transformer --checkpoint=20 --data ../processed_data/nuScenes_test_map_full.pkl --output_path results --output_tag ViT_transformer --node_type VEHICLE --prediction_horizon 12```|
| Lane_encoding_CNN |```python evaluate.py --model models/Lane_transformer --checkpoint=20 --data ../processed_data/nuScenes_test_lane_full.pkl --output_path results --output_tag Lane_transformer --node_type VEHICLE --prediction_horizon 12```|

# TODO
- [x] Rename Folders and command.
- [x] Debug Lane preprocess.
- [x] Coding plot code for evaluation.
- [x] Update the data link with new zip file.
- [x] Update the requirment txt to the latest version.
- [ ] Update the python version.
- [x] Update the code structure : 
    - [x] Remove unused part.(Pedestrian)
- [ ] Integrate Unit test to repo.
