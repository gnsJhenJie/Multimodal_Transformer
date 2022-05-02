import torch
from torch import nn, optim, utils
import numpy as np
import os
import time
import dill
import json
import random
import pathlib
import warnings
from tqdm import tqdm
import visualization
import evaluation
import matplotlib.pyplot as plt
from argument_parser import args
from model.trajectron import Trajectron
from model.model_registrar import ModelRegistrar
from model.model_utils import cyclical_lr
from model.dataset import EnvironmentDataset, collate
from tensorboardX import SummaryWriter
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def main():
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        print('Config json not found!')
    with open(args.conf, 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['incl_robot_node'] = args.incl_robot_node
    hyperparams['node_freq_mult_train'] = args.node_freq_mult_train
    hyperparams['node_freq_mult_eval'] = args.node_freq_mult_eval
    hyperparams['scene_freq_mult_train'] = args.scene_freq_mult_train
    hyperparams['scene_freq_mult_eval'] = args.scene_freq_mult_eval
    hyperparams['scene_freq_mult_viz'] = args.scene_freq_mult_viz
    hyperparams['edge_encoding'] = False
    hyperparams['map_cnn_encoding'] = args.map_cnn_encoding
    hyperparams['map_vit_encoding'] = args.map_vit_encoding
    hyperparams['lane_cnn_encoding'] = args.lane_cnn_encoding
    hyperparams['augment'] = args.augment
    hyperparams['override_attention_radius'] = args.override_attention_radius
    hyperparams['autoregressive'] = args.autoregressive

    torch.autograd.set_detect_anomaly(True)
    # This is needed for memory pinning using a DataLoader (otherwise memory
    # is pinned to cuda:0 by default)
    torch.cuda.set_device(args.device)
    args.eval_device = args.device
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    print('-----------------------')
    print('| TRAINING PARAMETERS |')
    print('-----------------------')
    print('| batch_size: %d' % args.batch_size)
    print('| device: %s' % args.device)
    print('| eval_device: %s' % args.eval_device)
    print('| Learning rate %s' % hyperparams['learning_rate'])
    print('| Autoregressive mode  %s' % hyperparams['autoregressive'])
    if hyperparams['map_cnn_encoding']:
        print('| CNN encoding mode  %s' % hyperparams['map_cnn_encoding'])
    elif hyperparams['map_vit_encoding']:
        print('| Vision transformer encoding mode  %s' %
              hyperparams['map_vit_encoding'])
    elif hyperparams['lane_cnn_encoding']:
        print('| CNN encoding lane mode  %s' %
              hyperparams['lane_cnn_encoding'])
    else:
        print('| Basic mode (No Map Information Encoding)  True')
    print('| Data augment mode  %s' % hyperparams['augment'])
    print('| MHL: %s' % hyperparams['minimum_history_length'])
    print('| PH: %s' % hyperparams['prediction_horizon'])
    print('-----------------------')

    log_writer = None
    model_dir = None
    if not args.debug:
        # Create the log and model directiory if they're not present.
        model_dir = os.path.join(args.log_dir, args.log_tag)
        pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

        # Save config to model directory
        with open(os.path.join(model_dir, 'config.json'), 'w') as conf_json:
            json.dump(hyperparams, conf_json)

        log_writer = SummaryWriter(log_dir=model_dir)

    # Load training and evaluation environments and scenes
    train_scenes = []
    if hyperparams['lane_cnn_encoding']:
        train_data_name = args.data_name + "_train" + "_lane" + "_full.pkl"
    else:
        train_data_name = args.data_name + "_train" + "_map" + "_full.pkl"
    train_data_path = os.path.join(args.data_dir, train_data_name)
    with open(train_data_path, 'rb') as f:
        train_env = dill.load(f, encoding='latin1')

    for attention_radius_override in args.override_attention_radius:
        node_type1, node_type2, attention_radius = attention_radius_override.split(
            ' ')
        train_env.attention_radius[(node_type1, node_type2)] = float(
            attention_radius)

    if train_env.robot_type is None and hyperparams['incl_robot_node']:
        # TODO: Make more general, allow the user to specify?
        train_env.robot_type = train_env.NodeType[0]
        for scene in train_env.scenes:
            scene.add_robot_from_nodes(train_env.robot_type)

    train_scenes = train_env.scenes
    train_scenes_sample_probs = train_env.scenes_freq_mult_prop if args.scene_freq_mult_train else None

    train_dataset = EnvironmentDataset(
        train_env,
        hyperparams['state'],
        hyperparams['pred_state'],
        scene_freq_mult=hyperparams['scene_freq_mult_train'],
        node_freq_mult=hyperparams['node_freq_mult_train'],
        hyperparams=hyperparams,
        min_history_timesteps=hyperparams['minimum_history_length'],
        min_future_timesteps=hyperparams['prediction_horizon'],
        return_robot=not args.incl_robot_node)
    train_data_loader = dict()
    for node_type_data_set in train_dataset:
        node_type_dataloader = utils.data.DataLoader(
            node_type_data_set,
            collate_fn=collate,
            pin_memory=False if args.device == 'cpu' else True,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.preprocess_workers)
        train_data_loader[node_type_data_set.node_type] = node_type_dataloader

    print(f"Loaded training data from {train_data_path}")

    eval_scenes = []
    eval_scenes_sample_probs = None
    if args.eval_every is not None:
        if hyperparams['lane_cnn_encoding']:
            test_data_name = args.data_name + "_val" + "_lane" + "_full.pkl"
        else:
            test_data_name = args.data_name + "_val" + "_map" + "_full.pkl"
        eval_data_path = os.path.join(args.data_dir, test_data_name)
        with open(eval_data_path, 'rb') as f:
            eval_env = dill.load(f, encoding='latin1')

        for attention_radius_override in args.override_attention_radius:
            node_type1, node_type2, attention_radius = attention_radius_override.split(
                ' ')
            eval_env.attention_radius[(node_type1, node_type2)] = float(
                attention_radius)

        if eval_env.robot_type is None and hyperparams['incl_robot_node']:
            # TODO: Make more general, allow the user to specify?
            eval_env.robot_type = eval_env.NodeType[0]
            for scene in eval_env.scenes:
                scene.add_robot_from_nodes(eval_env.robot_type)

        eval_scenes = eval_env.scenes
        eval_scenes_sample_probs = eval_env.scenes_freq_mult_prop if args.scene_freq_mult_eval else None

        eval_dataset = EnvironmentDataset(
            eval_env,
            hyperparams['state'],
            hyperparams['pred_state'],
            scene_freq_mult=hyperparams['scene_freq_mult_eval'],
            node_freq_mult=hyperparams['node_freq_mult_eval'],
            hyperparams=hyperparams,
            min_history_timesteps=hyperparams['minimum_history_length'],
            min_future_timesteps=hyperparams['prediction_horizon'],
            return_robot=not args.incl_robot_node)
        eval_data_loader = dict()
        for node_type_data_set in eval_dataset:
            node_type_dataloader = utils.data.DataLoader(
                node_type_data_set,
                collate_fn=collate,
                pin_memory=False if args.eval_device == 'cpu' else True,
                batch_size=args.eval_batch_size,
                shuffle=True,
                num_workers=args.preprocess_workers)
            eval_data_loader[node_type_data_set.node_type] = node_type_dataloader

        print(f"Loaded evaluation data from {eval_data_path}")

    model_registrar = ModelRegistrar(model_dir, args.device)

    trajectron = Trajectron(model_registrar,
                            hyperparams,
                            log_writer,
                            args.device)

    trajectron.set_environment(train_env)

    print('Created Training Model.')

    eval_trajectron = None
    if args.eval_every is not None or args.vis_every is not None:
        eval_trajectron = Trajectron(model_registrar,
                                     hyperparams,
                                     log_writer,
                                     args.eval_device)
        eval_trajectron.set_environment(eval_env)

    print('Created Evaluation Model.')

    optimizer = dict()
    lr_scheduler = dict()
    for node_type in train_env.NodeType:
        if node_type not in hyperparams['pred_state']:
            continue
        optimizer[node_type] = optim.AdamW([{'params': model_registrar.get_all_but_name_match('map_encoder').parameters()}, {
                                           'params': model_registrar.get_name_match('map_encoder').parameters(), 'lr': 0.0008}], lr=hyperparams['learning_rate'])
        # print("\nmodel_parameter : ",model_registrar.model_dict)
        # Set Learning Rate
        if hyperparams['learning_rate_style'] == 'const':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=1.0)
        elif hyperparams['learning_rate_style'] == 'exp':
            lr_scheduler[node_type] = optim.lr_scheduler.ExponentialLR(
                optimizer[node_type], gamma=hyperparams['learning_decay_rate'])

    #################################
    #           TRAINING            #
    #################################
    curr_iter_node_type = {
        node_type: 0 for node_type in train_data_loader.keys()}
    for epoch in range(1, args.train_epochs + 1):
        model_registrar.to(args.device)
        train_dataset.augment = args.augment
        for node_type, data_loader in train_data_loader.items():
            if node_type == "PEDESTRIAN":
                continue
            curr_iter = curr_iter_node_type[node_type]
            pbar = tqdm(data_loader, ncols=80)
            for batch in pbar:
                trajectron.set_curr_iter(curr_iter)
                optimizer[node_type].zero_grad()
                train_loss, reg_loss = trajectron.train_loss(batch, node_type)
                pbar.set_description(
                    f"Epoch {epoch}, {node_type} L: {reg_loss.item()*80:.4f}")
                train_loss.backward()
                # Clipping gradients.
                if hyperparams['grad_clip'] is not None:
                    nn.utils.clip_grad_value_(
                        model_registrar.parameters(), hyperparams['grad_clip'])
                optimizer[node_type].step()

                # Stepping forward the learning rate scheduler and annealers.
                lr_scheduler[node_type].step()
                if not args.debug:
                    log_writer.add_scalar(
                        f"{node_type}/train/learning_rate",
                        lr_scheduler[node_type].get_last_lr()[0],
                        curr_iter)
                    log_writer.add_scalar(
                        f"{node_type}/train/loss", train_loss, curr_iter)

                curr_iter += 1
            curr_iter_node_type[node_type] = curr_iter
        train_dataset.augment = False
        if args.eval_every is not None or args.vis_every is not None:
            eval_trajectron.set_curr_iter(epoch)

        #################################
        #        VISUALIZATION          #
        #################################
        if args.vis_every is not None and not args.debug and epoch % args.vis_every == 0 and epoch > 0 and False:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            with torch.no_grad():
                # Predict random timestep to plot for train data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(
                        train_scenes, p=train_scenes_sample_probs)
                else:
                    scene = np.random.choice(train_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = trajectron.predict(scene,
                                                 timestep,
                                                 ph,
                                                 z_mode=True,
                                                 gmm_mode=True,
                                                 all_z_sep=False,
                                                 full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(
                    ax,
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('train/prediction', fig, epoch)

                model_registrar.to(args.eval_device)
                # Predict random timestep to plot for eval data set
                if args.scene_freq_mult_viz:
                    scene = np.random.choice(
                        eval_scenes, p=eval_scenes_sample_probs)
                else:
                    scene = np.random.choice(eval_scenes)
                timestep = scene.sample_timesteps(1, min_future_timesteps=ph)
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      num_samples=20,
                                                      min_future_timesteps=ph,
                                                      z_mode=False,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(
                    ax,
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction', fig, epoch)

                # Predict random timestep to plot for eval data set
                predictions = eval_trajectron.predict(scene,
                                                      timestep,
                                                      ph,
                                                      min_future_timesteps=ph,
                                                      z_mode=True,
                                                      gmm_mode=True,
                                                      all_z_sep=True,
                                                      full_dist=False)

                # Plot predicted timestep for random scene
                fig, ax = plt.subplots(figsize=(10, 10))
                visualization.visualize_prediction(
                    ax,
                    predictions,
                    scene.dt,
                    max_hl=max_hl,
                    ph=ph,
                    map=scene.map['VISUALIZATION'] if scene.map is not None else None)
                ax.set_title(f"{scene.name}-t: {timestep}")
                log_writer.add_figure('eval/prediction_all_z', fig, epoch)

        #################################
        #           EVALUATION          #
        #################################
        if args.eval_every is not None and not args.debug and epoch % args.eval_every == 0 and epoch > 0:
            max_hl = hyperparams['maximum_history_length']
            ph = hyperparams['prediction_horizon']
            model_registrar.to(args.eval_device)
            with torch.no_grad():
                # Calculate evaluation loss
                for node_type, data_loader in eval_data_loader.items():
                    if node_type == "PEDESTRIAN":
                        continue
                    eval_loss = []
                    print(
                        f"Starting Evaluation @ epoch {epoch} for node type: {node_type}")
                    pbar = tqdm(data_loader, ncols=80)
                    for batch in pbar:
                        eval_loss_node_type, reg_loss = eval_trajectron.eval_loss(
                            batch, node_type)
                        pbar.set_description(
                            f"Epoch {epoch}, {node_type} L: {reg_loss.item():.4f}")
                        eval_loss.append(
                            {node_type: {'nll': [eval_loss_node_type]}})
                        del batch

                    evaluation.log_batch_errors(eval_loss,
                                                log_writer,
                                                f"{node_type}/eval_loss",
                                                epoch)

                # Predict batch timesteps for evaluation dataset evaluation
                # eval_batch_errors = []
                # for scene in tqdm(eval_scenes, desc='Sample Evaluation', ncols=80):
                #     timesteps = scene.sample_timesteps(args.eval_batch_size)

                #     predictions = eval_trajectron.predict(scene,
                #                                           timesteps,
                #                                           ph,
                #                                           num_samples=50,
                #                                           min_future_timesteps=ph,
                #                                           full_dist=False)

                #     eval_batch_errors.append(evaluation.compute_batch_statistics(predictions,
                #                                                                  scene.dt,
                #                                                                  max_hl=max_hl,
                #                                                                  ph=ph,
                #                                                                  node_type_enum=eval_env.NodeType,
                # map=scene.map))

                # evaluation.log_batch_errors(eval_batch_errors,
                #                             log_writer,
                #                             'eval',
                #                             epoch,
                #                             bar_plot=['kde'],
                #                             box_plot=['ade', 'fde'])

                # # Predict maximum likelihood batch timesteps for evaluation dataset evaluation
                # eval_batch_errors_ml = []
                # for scene in tqdm(eval_scenes, desc='MM Evaluation', ncols=80):
                #     timesteps = scene.sample_timesteps(scene.timesteps)

                #     predictions = eval_trajectron.predict(scene,
                #                                           timesteps,
                #                                           ph,
                #                                           num_samples=1,
                #                                           min_future_timesteps=ph,
                #                                           z_mode=True,
                #                                           gmm_mode=True,
                #                                           full_dist=False)

                #     eval_batch_errors_ml.append(evaluation.compute_batch_statistics(predictions,
                #                                                                     scene.dt,
                #                                                                     max_hl=max_hl,
                #                                                                     ph=ph,
                #                                                                     map=scene.map,
                #                                                                     node_type_enum=eval_env.NodeType,
                # kde=False))

                # evaluation.log_batch_errors(eval_batch_errors_ml,
                #                             log_writer,
                #                             'eval/ml',
                #                             epoch)

        if args.save_every is not None and args.debug is False and epoch % args.save_every == 0:
            model_registrar.save_models(epoch)


if __name__ == '__main__':
    main()
