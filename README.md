# Implementation progress
https://hackmd.io/tcPg3HmMSC-PXi-XkTUdXA
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

Download nuScene dataset:

> ```wget --no-check-certificate -O dataset.zip"https://140.115.54.247/share.cgissid=2f7bb2e3f7ea4444903cee95da46207c&fid=2f7bb2e3f7ea4444903cee95da46207c&filename=data.zip&openfolder=forcedownload&ep="```
    
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
        `--offline_process
        
To preprocess the nuScenes data, you can execute the command below in `offline_process` directory, depending on the data version you desire.<br></br>
| Typo preprocess| Command|
|:---|:---|
| Map preprocess |```python3 nuScenes_process_data.py --data=../../data/sets/nuscenes/v1.0/ --version=v1.0-trainval --output_path=../processed_data'```|
| Lane preprocess |```python3 nuScenes_process_data.py --data=../../data/sets/nuscenes/v1.0/ --version=v1.0-trainval --output_path=../processed_data' --process_lane``` (not finish)|

## Training

    ${Trajectron root}
    ├── Transformer

To train a model with nuScene data, you can execute the command below in `Transformer` directory, depending on the model version you desire.<br></br>
| Model| Command|
|:---|:---|
| Basic |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/offline_process/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/offline_process/models --train_epochs 20 --node_freq_mult_train --log_tag _int_ee --augment```|
| Map_encoding_CNN |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/offline_process/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/offline_process/models --train_epochs 20 --node_freq_mult_train --log_tag _int_ee --augment```|
| Map_encoding_ViT |```python3 train.py --eval_every 1 --vis_every 1 --conf ../experiments/offline_process/config/config.json --data_name nuScenes --preprocess_workers 8 --batch_size 128 --log_dir ../experiments/offline_process/models --train_epochs 20 --node_freq_mult_train --log_tag _int_ee --augment```|

# Evaluate Progress
    
    ${Trajectron root}
    ├── experiments
        `-- offline_process
        
If you want to evaluate the trained model to generate trajectories and plot them, you can use `NuScenes Qualitative.ipynb` norebook.<br></br>
To evaluate a trained model's performance on forecasting vehicles, you can execute a one of the following commands from within the `experiments/nuScenes` directory.
    
| Model| Command|
|:---|:---|
| Basic |```python evaluate.py --model models/lane_transformer --checkpoint=15 --data ../processed_data/nuScenes_test_full.pkl --output_path results --output_tag lane_transformer --node_type VEHICLE --prediction_horizon 6```|
| Map_encoding_CNN |```python evaluate.py --model models/lane_transformer --checkpoint=15 --data ../processed_data/nuScenes_test_full.pkl --output_path results --output_tag lane_transformer --node_type VEHICLE --prediction_horizon 6```|
| Map_encoding_ViT |```python evaluate.py --model models/lane_transformer --checkpoint=15 --data ../processed_data/nuScenes_test_full.pkl --output_path results --output_tag lane_transformer --node_type VEHICLE --prediction_horizon 6```|

# TODO
- [ ] Rename Folders and command.
- [ ] Debug Lane preprocess.
- [ ] Coding plot code for evaluation.
- [ ] Update the data link with new zip file.
- [ ] Update the requirment txt to the latest version.
- [ ] Update the python version.
- [ ] Update the code structure : 
    - [ ] Remove unused part.(Pedestrian)
    - [ ] Rename program name.
    - [ ] Fix class structure.