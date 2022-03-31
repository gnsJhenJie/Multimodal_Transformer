import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--conf",
                    help="path to json config file for hyperparameters",
                    type=str,
                    default='../experiments/offline_process/models/int_ee/config.json')

parser.add_argument("--debug",
                    help="disable all disk writing processes.",
                    action='store_true')

parser.add_argument("--preprocess_workers",
                    help="number of processes to spawn for preprocessing",
                    type=int,
                    default=10)

# Model Parameters

parser.add_argument('--override_attention_radius',
                    action='append',
                    help='Specify one attention radius to override. E.g. "PEDESTRIAN VEHICLE 10.0"',
                    default=[])

parser.add_argument('--incl_robot_node',
                    help="whether to include a robot node in the graph or simply model all agents",
                    action='store_true')

parser.add_argument('--map_encoding',
                    help="Whether to use map encoding or not",
                    action='store_true')

parser.add_argument('--augment',
                    help="Whether to augment the scene during training",
                    action='store_true')

parser.add_argument('--node_freq_mult_train',
                    help="Whether to use frequency multiplying of nodes during training",
                    action='store_true')

parser.add_argument('--node_freq_mult_eval',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--scene_freq_mult_train',
                    help="Whether to use frequency multiplying of nodes during training",
                    action='store_true')

parser.add_argument('--scene_freq_mult_eval',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--scene_freq_mult_viz',
                    help="Whether to use frequency multiplying of nodes during evaluation",
                    action='store_true')

parser.add_argument('--no_edge_encoding',
                    help="Whether to use neighbors edge encoding",
                    action='store_true')

# Data Parameters
parser.add_argument("--data_dir",
                    help="what dir to look in for data",
                    type=str,
                    default='../experiments/processed_data/')

parser.add_argument("--data_name",
                    help="what file to load for training data",
                    type=str,
                    default='nuScenes')

# parser.add_argument("--train_data_dict",
#                     help="what file to load for training data",
#                     type=str,
#                     default='train.pkl')

# parser.add_argument("--eval_data_dict",
#                     help="what file to load for evaluation data",
#                     type=str,
#                     default='val.pkl')

parser.add_argument("--log_dir",
                    help="what dir to save training information (i.e., saved models, logs, etc)",
                    type=str,
                    default='../experiments/offline_process/models')

parser.add_argument("--log_tag",
                    help="tag for the log folder",
                    type=str,
                    default='_int_ee_lr_0.02_src_padding_mask_mean_2dim')

parser.add_argument('--device',
                    help='what device to perform training on',
                    type=str,
                    default='cuda:0')

parser.add_argument("--eval_device",
                    help="what device to use during evaluation",
                    type=str,
                    default=None)

# Training Parameters
parser.add_argument("--train_epochs",
                    help="number of iterations to train for",
                    type=int,
                    default=20)

parser.add_argument('--batch_size',
                    help='training batch size',
                    type=int,
                    default=256)

parser.add_argument('--eval_batch_size',
                    help='evaluation batch size',
                    type=int,
                    default=256)

parser.add_argument('--k_eval',
                    help='how many samples to take during evaluation',
                    type=int,
                    default=25)

parser.add_argument('--seed',
                    help='manual seed to use, default is 123',
                    type=int,
                    default=123)

parser.add_argument('--eval_every',
                    help='how often to evaluate during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--vis_every',
                    help='how often to visualize during training, never if None',
                    type=int,
                    default=1)

parser.add_argument('--save_every',
                    help='how often to save during training, never if None',
                    type=int,
                    default=1)
parser.add_argument("--autoregressive",
                    help="disable all disk writing processes.",
                    action='store_true')
args = parser.parse_args()
