from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../Transformer")

import utils
import evaluation
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model full path", type=str)
parser.add_argument(
    "--checkpoint", help="model checkpoint to evaluate", type=int)
parser.add_argument("--data", help="full path to data file", type=str)
parser.add_argument("--output_path", help="path to output csv file", type=str)
parser.add_argument("--output_tag", help="name tag for output file", type=str)
parser.add_argument("--node_type", help="node type to evaluate", type=str)
parser.add_argument("--prediction_horizon", nargs='+',
                    help="prediction horizon", type=int, default=None)
args = parser.parse_args()


def compute_road_violations(predicted_trajs, map, channel):
    obs_map = 1 - map.data[..., channel, :, :] / 255

    interp_obs_map = RectBivariateSpline(range(obs_map.shape[0]),
                                         range(obs_map.shape[1]),
                                         obs_map,
                                         kx=1, ky=1)

    old_shape = predicted_trajs.shape
    pred_trajs_map = map.to_map_points(predicted_trajs.reshape((-1, 2)))

    traj_obs_values = interp_obs_map(
        pred_trajs_map[:, 0], pred_trajs_map[:, 1], grid=False)
    traj_obs_values = traj_obs_values.reshape(
        (old_shape[0], old_shape[1], old_shape[2]))
    num_viol_trajs = np.sum(traj_obs_values.max(axis=2) > 0, dtype=float)

    return num_viol_trajs


def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    return trajectron, hyperparams


if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                ' ')
            env.attention_radius[(node_type1, node_type2)
                                 ] = float(attention_radius)

    scenes = env.scenes

    for ph in args.prediction_horizon:
        print(f"Prediction Horizon: {ph}")
        max_hl = hyperparams['maximum_history_length']

        with torch.no_grad():
            eval_ade_batch_errors = np.array([])
            eval_fde_batch_errors = np.array([])
            eval_kde_nll = np.array([])
            eval_road_viols = np.array([])
            print("-- Evaluating Full")
            for scene in tqdm(scenes):
                timesteps = np.arange(scene.timesteps)

                # change to our prediction result
                history_pred, lane_pred = eval_stg.predict(
                    scene, timesteps, ph, min_history_timesteps=hyperparams['minimum_history_length'], min_future_timesteps=ph)

                if not history_pred and not lane_pred:
                    continue

                eval_road_viols_batch = []

                if hyperparams['lane_cnn_encoding']:
                    lane_prediction_dict, _, _ = utils.lane_prediction_output_to_trajectories(
                        lane_pred, scene.dt, max_hl, ph, prune_ph_to_future=False)
                else:
                    history_prediction_dict, _, _ = utils.prediction_output_to_trajectories(
                        history_pred, scene.dt, max_hl, ph, prune_ph_to_future=False)

                    # check for multiple output offroad rate
                    # for t in history_prediction_dict.keys():
                    #     for node in history_prediction_dict[t].keys():
                    #         if node.type == args.node_type:
                    #             viols = compute_road_violations(history_prediction_dict[t][node],
                    #                                             scene.map[args.node_type],
                    #                                             channel=0)

                    #         # for lane in lane_prediction_dict[t][node].keys():
                    #         #     lane_viols = compute_road_violations(lane_prediction_dict[t][node][lane],
                    #         #                                 scene.map[args.node_type],
                    #         #                                 channel=0)
                    #         # print('viols : ', viols)

                    #         eval_road_viols_batch.append(viols)
                    # eval_road_viols = np.hstack(
                    #     (eval_road_viols, eval_road_viols_batch))
                    
                    if hyperparams['lane_cnn_encoding']:
                        batch_error_dict = evaluation.lane_compute_batch_statistics(
                            lane_pred,
                            scene.dt,
                            max_hl=max_hl,
                            ph=ph,
                            node_type_enum=env.NodeType,
                            map=None,
                            prune_ph_to_future=False)
                    else:
                        batch_error_dict = evaluation.compute_batch_statistics(
                            history_pred,
                            scene.dt,
                            max_hl=max_hl,
                            ph=ph,
                            node_type_enum=env.NodeType,
                            map=None,
                            prune_ph_to_future=False)

                    eval_ade_batch_errors = np.hstack(
                        (eval_ade_batch_errors, batch_error_dict[args.node_type]['ade']))
                    eval_fde_batch_errors = np.hstack(
                        (eval_fde_batch_errors, batch_error_dict[args.node_type]['fde']))
                    # eval_kde_nll = np.hstack((eval_kde_nll, batch_error_dict[args.node_type]['kde']))

        pd.DataFrame({'value': eval_ade_batch_errors, 'metric': 'ade', 'type': 'full'}).to_csv(
            os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_ade_full.csv'))
        pd.DataFrame({'value': eval_fde_batch_errors, 'metric': 'fde', 'type': 'full'}).to_csv(
            os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_fde_full.csv'))
        # # pd.DataFrame({'value': eval_kde_nll, 'metric': 'kde', 'type': 'full'}
        # #              ).to_csv(os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_kde_full.csv'))
        pd.DataFrame({'value': eval_road_viols, 'metric': 'road_viols', 'type': 'full'}).to_csv(
            os.path.join(args.output_path, args.output_tag + "_" + str(ph) + '_rv_full.csv'))
