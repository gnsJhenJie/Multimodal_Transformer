from model.components import Encoder, Trajectory_Decoder, Lane_Encoder, Lane_Decoder, CNNMapEncoder, Mlp
from model.components import *
from model.model_utils import L2_norm, obs_violation_rate, classification_loss, generate_square_subsequent_mask, generate_mask, rgetattr, rsetattr
from model.model_utils import *
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from vit_pytorch import ViT
from environment.scene_graph import DirectedEdge


class MultimodalGenerativeCVAE(object):
    def __init__(self,
                 env,
                 node_type,
                 model_registrar,
                 hyperparams,
                 device,
                 edge_types,
                 log_writer=None):

        self.hyperparams = hyperparams
        self.env = env
        self.node_type = node_type
        self.model_registrar = model_registrar
        self.log_writer = log_writer
        self.device = device
        self.edge_types = [
            edge_type for edge_type in edge_types if edge_type[0] is node_type]
        self.curr_iter = 0

        self.node_modules = dict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state'][node_type]
        self.state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.state[node_type].values()]))
        if self.hyperparams['incl_robot_node']:
            self.robot_state_length = int(
                np.sum([len(entity_dims)
                        for entity_dims in self.state[env.robot_type].values()])
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))

        edge_types_str = [DirectedEdge.get_str_from_types(
            *edge_type) for edge_type in self.edge_types]
        self.create_submodule(edge_types_str)

        self.memory = None
        self.memory_mask = None

    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model_if_absent):
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_models(self):
        #######################
        # Transformer Encoder #
        #######################

        self.add_submodule(self.node_type + '/node_history_encoder',
                           model_if_absent=Encoder(ninp=self.state_length,
                                                   nlayers=self.hyperparams['transformer']['nlayers'],
                                                   nhead=self.hyperparams['transformer']['nhead'],
                                                   in_dim=self.hyperparams['transformer']['in_dim'],
                                                   fdim=self.hyperparams['transformer']['fdim']))

        ###################
        #   Map Encoder   #
        ###################
        map_output_size = None
        if self.hyperparams['use_map_encoding']:

            if self.hyperparams['map_encoder']['transformer']:
                me_params = self.hyperparams['map_encoder']['vit_param']
                map_output_size = me_params['output_size']
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=ViT(
                                                    image_size = me_params['image_size'],
                                                    patch_size = me_params['patch_size'],
                                                    num_classes = me_params['output_size'],
                                                    dim = me_params['dim'],
                                                    depth = me_params['deep'],
                                                    heads = me_params['heads'],
                                                    mlp_dim = me_params['mlp_dim'],
                                                    dropout = me_params['dropout'],
                                                    emb_dropout = me_params['emb_dropout']))
            else:
                me_params = self.hyperparams['map_encoder']['cnn_param']
                map_output_size = me_params['output_size']
                self.add_submodule(self.node_type + '/map_encoder',
                                   model_if_absent=CNNMapEncoder(me_params['map_channels'],
                                                                 me_params['hidden_channels'],
                                                                 me_params['output_size'],
                                                                 me_params['masks'],
                                                                 me_params['strides'],
                                                                 me_params['patch_size']))
        elif self.hyperparams['lane_encoding']:
            ###################
            #   lane Encoder  #
            ###################
            if self.node_type in self.hyperparams['lane_encoder']:
                me_params = self.hyperparams['lane_encoder'][self.node_type]
                self.add_submodule(self.node_type + '/lane_encoder',
                                   model_if_absent=Lane_Encoder(me_params['nlayers'],
                                                                me_params['embedding_size'],
                                                                me_params['map_channels'],
                                                                me_params['output_size'],
                                                                me_params['kernel_size'],
                                                                me_params['strides']))
            ###################
            #   lane Decoder  #
            ###################
            if self.node_type in self.hyperparams['lane_decoder']:
                me_params = self.hyperparams['lane_decoder'][self.node_type]
                self.add_submodule(self.node_type + '/lane_decoder',
                                   model_if_absent=Lane_Decoder(nlayers=me_params['nlayers'],
                                                                ninp=self.hyperparams['lane_encoder'][self.node_type]['output_size'],
                                                                in_dim=me_params['in_dim'],
                                                                nhead=me_params['nhead'],
                                                                fdim=me_params['fdim'],
                                                                noutput=me_params['output_size']))
                ###################
                #  MLP + softmax  #
                ###################
                self.add_submodule(self.node_type + '/Lane/MLP_Softmax',
                                   model_if_absent=Mlp(in_channels=me_params['output_size']*self.hyperparams['max_lane_num'],
                                                       output_size=self.hyperparams['max_lane_num'],
                                                       layer_num=self.hyperparams['mlp_layer'],
                                                       mode='classification'))
        # Node History Encoder
        if self.hyperparams['use_map_encoding']:
            fusion_layer_size = self.hyperparams['transformer']['output_size'] + map_output_size
            self.add_submodule(self.node_type+'/fusion/hist_map',
                               model_if_absent=Mlp(in_channels=fusion_layer_size,
                                                   output_size=self.hyperparams['transformer']['output_size'],
                                                   layer_num=self.hyperparams['fusion_hist_map_layer'],
                                                   mode='regression'))
        #######################
        # Transformer Decoder #
        #######################
        self.add_submodule(self.node_type + '/decoder/transformer_decoder',
                           model_if_absent=Trajectory_Decoder(nlayers=self.hyperparams['transformer']['nlayers'],
                                                              tgt_inp=self.pred_state_length,
                                                              lane_inp=self.hyperparams['lane_encoder']['VEHICLE']['output_size'],
                                                              in_dim=self.hyperparams['transformer']['in_dim'],
                                                              nhead=self.hyperparams['transformer']['nhead'],
                                                              fdim=self.hyperparams['transformer']['fdim'],
                                                              noutput=self.hyperparams['transformer']['output_size']))

        ###################
        #   Decoder MLP   #
        ###################
        self.add_submodule(self.node_type + '/decoder/MLP',
                           model_if_absent=Mlp(in_channels=self.hyperparams['transformer']['output_size'],
                                               output_size=self.pred_state_length,
                                               layer_num=self.hyperparams['mlp_layer'],
                                               mode='regression'))

    def create_submodule(self, edge_types):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def obtain_encoded_tensors(self,
                               mode,
                               inputs,
                               inputs_st,
                               inputs_lane,
                               map):
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: tensor of Map information. [bs, channels, x, y]
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        candidate_lane = inputs_lane

        batch_size = inputs.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_pos = inputs[:, -1, 0:2]
        node_vel = inputs[:, -1, 2:4]

        node_history_st = inputs_st
        node_present_state_st = inputs_st[:, -1]

        ##################
        # Encode History #
        ##################
        memory_padding_mask = generate_mask(node_history_st).to(self.device)
        memory_src_mask = generate_square_subsequent_mask(node_history_st.size()[-2], self.device)
        # src , sort_order = cluster_input(src,mask)
        transformer_encoder = self.node_modules[self.node_type + '/node_history_encoder']
        memory = transformer_encoder(node_history_st, memory_src_mask, memory_padding_mask)

        ################
        # Map Encoding #
        ################
        encoded_map = None
        if self.hyperparams['use_map_encoding']:
            if self.hyperparams['map_encoder']['transformer']:
                encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1.)
            else:
                encoded_map = self.node_modules[self.node_type + '/map_encoder'](map * 2. - 1., (mode == ModeKeys.TRAIN))
                do = self.hyperparams['map_encoder']['cnn_param']['dropout']
                encoded_map = F.dropout(encoded_map, do, training=(mode == ModeKeys.TRAIN))
        #################
        # Lane Encoding #
        #################
        encoded_lane = None
        if self.hyperparams['lane_encoding'] and self.node_type in self.hyperparams['lane_encoder'] and self.hyperparams['lane_loss']:
            lane_num = self.hyperparams['max_lane_num']
            sample_num = self.hyperparams['sample_num']
            embedding_size = self.hyperparams['lane_encoder'][self.node_type]['embedding_size']
            candidate_lane = candidate_lane.view(
                batch_size*lane_num, embedding_size, sample_num)
            encoded_lane = self.node_modules[self.node_type + '/lane_encoder'](candidate_lane).view(
                batch_size, lane_num, self.hyperparams['lane_encoder'][self.node_type]['output_size'])

        return memory, memory_src_mask, memory_padding_mask, node_present_state_st, node_pos, encoded_lane, encoded_map

    def non_at_dec(self,
                   memory,
                   memory_mask,
                   memory_key_padding_mask,
                   labels_st,
                   n_s_t0,
                   lane_feature,
                   map_feature,
                   prediction_horizon):

        history_timestep = memory.size()[-2]
        init_pos = n_s_t0[:, 0:2]
        batch_size, pred_state = init_pos.size()

        transformer_decoder = self.node_modules[self.node_type + "/decoder/transformer_decoder"]
        mlp = self.node_modules[self.node_type + '/decoder/MLP']

        # get initial condition (training stage)
        tgt = torch.zeros([batch_size, 1, pred_state], device=self.device)
        tgt = torch.cat([tgt, labels_st[:, :-1, :]], dim=-2)

        hist_pred = None
        lane_pred = None
        # inference model by lane feature and history feature
        if self.hyperparams['lane_loss']:
            lane_pred = torch.tensor([], device=self.device)
            max_lane_num = lane_feature.size()[-2]
            for i in range(max_lane_num):
                tgt_mask = generate_square_subsequent_mask(tgt.size()[-2], self.device)
                h_state = transformer_decoder(tgt=tgt,
                                              memory=memory,
                                              tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              memory_key_padding_mask=memory_key_padding_mask,
                                              lane_feature=lane_feature[:, i:i+1, :].repeat(1, history_timestep, 1))
                lane_output = mlp(h_state).cumsum(dim=-2)
                lane_pred = torch.cat([lane_pred, lane_output.view(batch_size, 1, prediction_horizon, pred_state)], dim=-3)
        # inference model by only history feature
        else:
            tgt_mask = generate_square_subsequent_mask(tgt.size()[-2], self.device)
            h_state = transformer_decoder(tgt=tgt,
                                          memory=memory,
                                          tgt_mask=tgt_mask,
                                          memory_mask=memory_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
            hist_pred = mlp(h_state).cumsum(dim=-2)

        return hist_pred, lane_pred

    def at_dec(self,
               memory,
               memory_mask,
               memory_key_padding_mask,
               n_s_t0,
               lane_feature,
               map_feature,
               prediction_horizon):

        history_timestep = memory.size()[-2]
        init_pos = n_s_t0[:, 0:2]
        batch_size = init_pos.size()[0]
        pred_state = init_pos.size()[1]

        transformer_decoder = self.node_modules[self.node_type + "/decoder/transformer_decoder"]
        fusion = self.node_modules[self.node_type + '/fusion/hist_map']
        mlp = self.node_modules[self.node_type + '/decoder/MLP']

        lane_pred = None
        history_pred = None

        # inference model by lane feature and history feature
        if self.hyperparams['lane_loss']:
            lane_input = init_pos.view(batch_size, 1, pred_state)
            lane_pred = torch.tensor([], device=self.device)
            max_lane_num = lane_feature.size()[-2]
            for i in range(max_lane_num):
                lane_input = init_pos.view(batch_size, 1, pred_state)
                for _ in range(prediction_horizon):
                    tgt_mask = generate_square_subsequent_mask(lane_input.size()[-2], self.device)
                    h_state = transformer_decoder(tgt=lane_input,
                                                  memory=memory,
                                                  tgt_mask=tgt_mask,
                                                  memory_mask=memory_mask,
                                                  memory_key_padding_mask=memory_key_padding_mask,
                                                  lane_feature=lane_feature[:, i:i+1, :].repeat(1, history_timestep, 1))[:, -1, :]
                    delta_pos = mlp(h_state)
                    new_state = lane_input[:, -1, :] + delta_pos
                    lane_input = torch.cat([lane_input, new_state.view(batch_size, 1, pred_state)], dim=-2)
                lane_pred = torch.cat([lane_pred, lane_input.view(batch_size, 1, prediction_horizon+1, pred_state)], dim=-3)
            lane_pred = lane_pred[:, :, 1:, :]
        # inference model by only history feature
        else:
            history_input = init_pos.view(batch_size, 1, pred_state)
            for _ in range(prediction_horizon):
                tgt_mask = generate_square_subsequent_mask(history_input.size()[-2], self.device)
                h_state = transformer_decoder(tgt=history_input,
                                              memory=memory,
                                              tgt_mask=tgt_mask,
                                              memory_mask=memory_mask,
                                              memory_key_padding_mask=memory_key_padding_mask)[:, [-1], :]
                if self.hyperparams['use_map_encoding']:
                    h_state = fusion(torch.cat([h_state, map_feature.unsqueeze(-2)], dim=-1))
                new_state = mlp(h_state) + history_input[:, [-1], :]
                history_input = torch.cat([history_input, new_state], dim=-2)
                # print('history_input : ', history_input[0])
            history_pred = history_input[:, 1:, :]

        return history_pred, lane_pred

    def train_decoder(self,
                      autoregressive,
                      memory,
                      memory_mask,
                      memory_key_padding_mask,
                      labels_st,
                      n_s_t0,
                      encoded_lane,
                      encoded_map,
                      prediction_horizon):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        if autoregressive:
            history_pred, lane_pred = self.at_dec(memory,
                                                  memory_mask,
                                                  memory_key_padding_mask,
                                                  n_s_t0,
                                                  encoded_lane,
                                                  encoded_map,
                                                  prediction_horizon)
        else:
            history_pred, lane_pred = self.non_at_dec(memory,
                                                      memory_mask,
                                                      memory_key_padding_mask,
                                                      labels_st,
                                                      n_s_t0,
                                                      encoded_lane,
                                                      encoded_map,
                                                      prediction_horizon)
        # add mask inside the softmax
        pred_lane_index = self.classification_lane(encoded_lane)

        return history_pred, lane_pred, pred_lane_index

    def test_decoder(self,
                     memory,
                     memory_mask,
                     memory_key_padding_mask,
                     n_s_t0,
                     encoded_lane,
                     encoded_map,
                     prediction_horizon):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        history_pred, lane_pred = self.at_dec(memory,
                                              memory_mask,
                                              memory_key_padding_mask,
                                              n_s_t0,
                                              encoded_lane,
                                              encoded_map,
                                              prediction_horizon)
        # add mask inside the softmax
        pred_lane_index = self.classification_lane(encoded_lane)

        return history_pred, lane_pred, pred_lane_index

    def classification_lane(self, encoded_lane):

        pred_lane_index = None
        if self.hyperparams['lane_loss']:
            mlp = self.node_modules[self.node_type + '/Lane/MLP_Softmax']
            pred_lane_index = mlp(encoded_lane.view(encoded_lane.size()[0], encoded_lane.size()[1]*encoded_lane.size()[2]))

        return pred_lane_index

    def train_loss(self,
                   inputs,
                   inputs_st,
                   inputs_lane,
                   lane_label,
                   lane_mask,
                   lane_t_mask,
                   labels,
                   labels_st,
                   map,
                   prediction_horizon) -> torch.tensor:

        mode = ModeKeys.TRAIN
        memory, memory_src_mask, memory_key_padding_mask, n_s_t0, rel_state, encoded_lane, encoded_map = self.obtain_encoded_tensors(mode=mode,
                                                                                                                                     inputs=inputs,
                                                                                                                                     inputs_st=inputs_st,
                                                                                                                                     inputs_lane=inputs_lane,
                                                                                                                                     map=map)

        history_pred, lane_pred, lane_attn = self.train_decoder(self.hyperparams['autoregressive'],
                                                                memory,
                                                                memory_src_mask,
                                                                memory_key_padding_mask,
                                                                labels_st,
                                                                n_s_t0,
                                                                encoded_lane,
                                                                encoded_map,
                                                                prediction_horizon)
        # using hyperparam to choose loss mode
        reg_loss = 0
        cls_loss = 0
        con_loss = 0
        viol_rate = 0
        if self.hyperparams['lane_loss']:
            lane_reg_loss = L2_norm(lane_pred, labels_st.unsqueeze(1).repeat(1, 3, 1, 1))
            lane_reg_loss = torch.sum(lane_reg_loss, dim=-1)
            lane_reg_loss = torch.min(lane_reg_loss, dim=-1)[0]  # [nbs]
            dummy_loss = torch.zeros(lane_reg_loss.size(), dtype=torch.float, device=self.device)
            reg_loss = torch.where(lane_mask, lane_reg_loss, dummy_loss)
            # classification loss
            # cls_loss = classification_loss(lane_t_mask,
            #                                lane_mask,
            #                                lane_label,
            #                                lane_attn)
            # cls_loss = cls_loss / torch.sum(lane_t_mask)
            # dummy_loss = torch.zeros(cls_loss.size(), dtype=torch.float, device=self.device)
            # confidence loss
            # con_loss = confidence_loss(lane_label,lane_attn)
            # con_loss = torch.where(lane_mask, con_loss, dummy_loss)
        else:
            history_reg_loss = L2_norm(history_pred, labels_st)
            history_reg_loss = torch.sum(history_reg_loss, dim=-1) / prediction_horizon  # [nbs]
            lane_mask = torch.ones(history_reg_loss.size(), device=self.device)
            reg_loss = history_reg_loss
        reg_loss = torch.sum(reg_loss) / torch.sum(lane_mask)
        
        # can't deal with data augment
        # if self.hyperparams['use_map_encoding']:
        #     node_pos = inputs[:, [-1], 0:2]
        #     bs, _, feature_dim = node_pos.size()
        #     history_pred = history_pred*80 + node_pos
        #     # for loop calc
        #     for i in range(bs):
        #         viol_rate = viol_rate + obs_violation_rate(history_pred[[i],:,:], labels[[i],:,:], scene_map[i],heading_angle, 0, self.device)

        loss = reg_loss + cls_loss + con_loss

        if self.log_writer is not None:
            self.log_writer.add_scalar('%s/%s' % (str(self.node_type), 'loss'),
                                       loss,
                                       self.curr_iter)

        return loss

    def eval_loss(self,
                  node_type,
                  inputs,
                  inputs_st,
                  inputs_lane,
                  lane_label,
                  lane_mask,
                  lane_t_mask,
                  labels,
                  labels_st,
                  map,
                  prediction_horizon) -> torch.tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL
        memory, memory_mask, memory_key_padding_mask, n_s_t0, rel_state, encoded_lane, encoded_map = self.obtain_encoded_tensors(mode=mode,
                                                                                                                                 inputs=inputs,
                                                                                                                                 inputs_st=inputs_st,
                                                                                                                                 inputs_lane=inputs_lane,
                                                                                                                                 map=map)

        history_pred, lane_pred, lane_attn = self.test_decoder(memory,
                                                               memory_mask,
                                                               memory_key_padding_mask,
                                                               n_s_t0,
                                                               encoded_lane,
                                                               encoded_map,
                                                               prediction_horizon)
        # using hyperparam to choose loss mode
        reg_loss = 0
        if self.hyperparams['lane_loss']:
            lane_reg_loss = L2_norm(
                lane_pred, labels_st.unsqueeze(1).repeat(1, 3, 1, 1))
            lane_reg_loss = torch.sum(lane_reg_loss, dim=-1)
            lane_reg_loss = torch.min(lane_reg_loss, dim=-1)[0]  # [nbs]
            dummy_loss = torch.zeros(lane_reg_loss.size(), dtype=torch.float, device=self.device)
            reg_loss = torch.where(lane_mask, lane_reg_loss, dummy_loss)
        else:
            node_pos = inputs[:, [-1], 0:2]
            history_pred = history_pred*80 + node_pos
            history_reg_loss = L2_norm(history_pred, labels)
            history_reg_loss = torch.sum(history_reg_loss, dim=-1) / prediction_horizon   # [nbs]
            lane_mask = torch.ones(history_reg_loss.size(), dtype=torch.bool)
            reg_loss = history_reg_loss

        reg_loss = torch.sum(reg_loss) / torch.sum(lane_mask)

        cls_loss = 0
        con_loss = 0
        if self.hyperparams['lane_loss']:
            cls_loss = classification_loss(
                lane_t_mask, lane_mask, lane_label, lane_attn)
            cls_loss = cls_loss / torch.sum(lane_t_mask)
            # dummy_loss = torch.zeros(cls_loss.size(), dtype=torch.float, device=self.device)
            # confidence loss
            # con_loss = confidence_loss(lane_label,lane_attn)
            # con_loss = torch.where(lane_mask, con_loss, dummy_loss)

        loss = reg_loss + cls_loss + con_loss

        return loss

    def predict(self,
                inputs,
                inputs_st,
                inputs_lane,
                lane_mask,
                lane_t_mask,
                map,
                prediction_horizon):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        node_pos = inputs[:, [-1], 0:2]  # get init state
        bs, _, feature_dim = node_pos.size()
        mode = ModeKeys.EVAL

        memory, memory_mask, memory_key_padding_mask, n_s_t0, rel_state, encoded_lane, encoded_map = self.obtain_encoded_tensors(mode=mode,
                                                                                                                                 inputs=inputs,
                                                                                                                                 inputs_st=inputs_st,
                                                                                                                                 inputs_lane=inputs_lane,
                                                                                                                                 map=map)
        history_pred, lane_pred, _ = self.test_decoder(memory,
                                                        memory_mask,
                                                        memory_key_padding_mask,
                                                        n_s_t0,
                                                        encoded_lane,
                                                        encoded_map,
                                                        prediction_horizon)
        if self.hyperparams['lane_loss']:
            max_lane_num = lane_pred.size()[1]
            lane_pred = lane_pred*80 + node_pos.view(bs, 1, 1, feature_dim).repeat(1, max_lane_num, 1, 1)
        else:
            history_pred = history_pred*80 + node_pos

        return history_pred, lane_pred
