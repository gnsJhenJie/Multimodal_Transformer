import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE
from model.dataset import get_timesteps_data, restore


class Trajectron(object):
    def __init__(self, model_registrar,
                 hyperparams, log_writer,
                 device):
        super(Trajectron, self).__init__()

        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self.curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = dict()
        self.nodes = set()

        self.env = None

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[state_type].values()])
            )
        self.pred_state = self.hyperparams['pred_state']

    def set_environment(self, env):
        self.env = env

        self.node_models_dict.clear()
        edge_types = env.get_edge_types()

        for node_type in env.NodeType:
            # Only add a Model for NodeTypes we want to predict
            if node_type == "PEDESTRIAN":
                continue
            if node_type in self.pred_state.keys():
                self.node_models_dict[node_type] = MultimodalGenerativeCVAE(env,
                                                                            node_type,
                                                                            self.model_registrar,
                                                                            self.hyperparams,
                                                                            self.device,
                                                                            edge_types,
                                                                            log_writer=self.log_writer)

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter


    def train_loss(self, batch, node_type):

        (_,
         x_t, y_t, x_st_t, y_st_t,
         _,
         _,
         robot_traj_st_t,
         lanes, map) = batch

        # Turn lane data into tensor
        # [batch_size, lane_num, length, feature_dim]
        if self.hyperparams['lane_loss']:
            lane_mask = torch.tensor(lanes[0], dtype=torch.bool, device=self.device)
            # [batch_size, lane_num, 1] one-hot encoding
            lane_input = torch.tensor(np.array(lanes[1]), device=self.device, dtype=torch.float64)
            lane_label = torch.stack(lanes[2]).to(self.device)
            lane_t_mask = torch.stack(lanes[3]).to(self.device)
        else:
            lane_mask = None
            lane_input = None
            lane_label = None
            lane_t_mask = None
            
            
        x = x_t.to(self.device)
        y = y_t.to(self.device)

        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.train_loss(inputs=x,
                                inputs_st=x_st_t,
                                inputs_lane=lane_input,
                                lane_label=lane_label,
                                lane_mask=lane_mask,
                                lane_t_mask=lane_t_mask,
                                labels=y,
                                labels_st=y_st_t,
                                map=map,
                                prediction_horizon=self.ph)

        return loss

    def eval_loss(self, batch, node_type):

        (_,
         x_t, y_t, x_st_t, y_st_t,
         _,
         _,
         robot_traj_st_t,
         lanes, map) = batch

        lane_mask = None
        lane_input = None
        lane_label = None
        lane_t_mask = None
        x = x_t.to(self.device)
        y = y_t.to(self.device)

        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)
        if robot_traj_st_t is not None:
            robot_traj_st_t = robot_traj_st_t.to(self.device)
        if type(map) == torch.Tensor:
            map = map.to(self.device)

        # Run forward pass
        model = self.node_models_dict[node_type]
        loss = model.eval_loss(node_type,
                               inputs=x,
                               inputs_st=x_st_t,
                               inputs_lane=lane_input,
                               lane_label=lane_label,
                               lane_mask=lane_mask,
                               lane_t_mask=lane_t_mask,
                               labels=y,
                               labels_st=y_st_t,
                               map=map,
                               prediction_horizon=self.ph)

        return loss.cpu().detach().numpy()

    def predict(self,
                scene,
                timesteps,
                ph,
                min_future_timesteps=0,
                min_history_timesteps=1):

        history_pred_dict = {}
        lane_pred_dict = {}
        for node_type in self.env.NodeType:
            if node_type not in self.pred_state:
                continue

            model = self.node_models_dict[node_type]

            # Get Input data for node type and given timesteps
            batch = get_timesteps_data(env=self.env, scene=scene, t=timesteps, node_type=node_type, state=self.state,
                                       pred_state=self.pred_state, edge_types=model.edge_types,
                                       min_ht=min_history_timesteps, max_ht=self.max_ht, min_ft=min_future_timesteps,
                                       max_ft=min_future_timesteps, hyperparams=self.hyperparams)
            # There are no nodes of type present for timestep
            if batch is None:
                continue
            
            (_,
            x_t, y_t, x_st_t, y_st_t,
            _,
            _,
            robot_traj_st_t,
            lanes, map), nodes, timesteps_o = batch
            # print('lanes : ',lanes[2])

            # Turn lane data into tensor
            # [batch_size, lane_num, length, feature_dim]
            # lane_mask = torch.tensor(
            #     lanes[0], dtype=torch.bool, device=self.device)
            # # [batch_size, lane_num, 1] one-hot encoding
            # lane_input = torch.tensor(
            #     np.array(lanes[1]), device=self.device, dtype=torch.float)
            # lane_label = torch.stack(lanes[2]).to(self.device)
            # lane_t_mask = torch.stack(lanes[3]).to(self.device)
            lane_mask = None
            lane_input = None
            lane_label = None
            lane_t_mask = None

            x = x_t.to(self.device)
            x_st_t = x_st_t.to(self.device)
            if robot_traj_st_t is not None:
                robot_traj_st_t = robot_traj_st_t.to(self.device)
            if type(map) == torch.tensor:
                map = map.to(self.device)

            # Run forward pass
            history_pred, lane_pred = model.predict(inputs=x,
                                                    inputs_st=x_st_t,
                                                    inputs_lane=lane_input,
                                                    lane_mask=lane_mask,
                                                    lane_t_mask=lane_t_mask,
                                                    map=map,
                                                    prediction_horizon=ph)

            if self.hyperparams['lane_loss']: 
                max_lane_num = lane_pred.size()[1]
                # [bs, lane_num, timestep, feature_dim] -> [1, bs, lane_num, timestep, feature_dim]
                lane_pred = lane_pred.unsqueeze(0).cpu().detach().numpy()
                for i, ts in enumerate(timesteps_o):
                    if ts not in lane_pred_dict.keys():
                        lane_pred_dict[ts] = dict()
                    if nodes[i] not in lane_pred_dict[ts].keys():
                        lane_pred_dict[ts][nodes[i]] = dict()
                    for lane_index in range(max_lane_num):
                        lane_pred_dict[ts][nodes[i]][lane_index] = np.transpose(
                            lane_pred[:, [i], lane_index], (1, 0, 2, 3))
            else:
                # [bs, timestep, feature_dim] -> [1, bs, timestep, feature_dim]
                history_pred = history_pred.unsqueeze(0).cpu().detach().numpy()
                for i, ts in enumerate(timesteps_o):
                    if ts not in history_pred_dict.keys():
                        history_pred_dict[ts] = dict()
                    history_pred_dict[ts][nodes[i]] = np.transpose(
                        history_pred[:, [i]], (1, 0, 2, 3))

        return history_pred_dict, lane_pred_dict
