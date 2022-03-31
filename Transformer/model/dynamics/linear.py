from .dynamic import *
import torch

class Linear(Dynamic):

    def init_constants(self):
        pass

    def update_state(self, vt_control, idx):
        r"""
        dict shape [bs,2]
        Input a batch of control_unit to update_state
        we will use (x, y, phi, v) those value to represent object state
        :param: vt_control -> tensor [bs,2]
        :return: the next predict coordinate tensor [bs,2]
        """
        if self.curr_conditions == None:
            # [bs,2]
            p = self.initial_conditions['pos']
            v_0 = self.initial_conditions['vel']
            v = torch.norm(v_0,dim=1,p=2)
            phi = torch.atan2(v_0[:,0], v_0[:,1])
            self.curr_conditions = dict()
            self.curr_conditions['pos']=dict()
            self.curr_conditions['vel']=dict()
            self.curr_conditions['phi']=dict()
            
        else:
            # [bs,2,1]
            p = self.curr_conditions['pos'][idx]
            # [bs,1]
            v = self.curr_conditions['vel'][idx]
            phi = self.curr_conditions['phi'][idx]

        # print("\np size : ",p.size())
        # print("v size : ",v.size())
        # print("phi size : ",phi.size())
        # x_p = p[:,0]
        # y_p = p[:,1]
        # x_delta = vt_control[:,0]
        # y_delta = vt_control[:,1]
        
        
        # print("x_p size : ",x_p.size())
        # print("y_p size : ",y_p.size())
        # print("x_delta size : ",x_delta.size())
        # print("y_delta size : ",y_delta.size())
        
        ##############
        # version 1  #
        ##############
        x_p = p[:,0]
        y_p = p[:,1]
        dphi = vt_control[:,0]
        a = vt_control[:,1]
        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1
        mask = mask.view(mask.size()[0],1)
        # print("mask size : ",mask.size())

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        x_hat_1 = x_p + (a / dphi) * dcos_domega + v * dsin_domega + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt
        y_hat_1 = y_p - v * dcos_domega + (a / dphi) * dsin_domega - (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt
        x_1 = torch.cat([x_hat_1.unsqueeze(1),y_hat_1.unsqueeze(1)],dim=1)
        phi_hat_1 = phi + dphi * self.dt

        x_hat_2 = x_p + v * torch.cos(phi) * self.dt + (a / 2) * torch.cos(phi) * self.dt ** 2
        y_hat_2 = y_p + v * torch.sin(phi) * self.dt + (a / 2) * torch.sin(phi) * self.dt ** 2
        x_2 = torch.cat([x_hat_2.unsqueeze(1),y_hat_2.unsqueeze(1)],dim=1)
        phi_hat_2 = phi * torch.ones_like(a)
        v = v + a * self.dt

        # print("x_1 : ",x_1.size())
        # print("x_2 : ",x_2.size())
        # print("phi_hat_1 : ",phi_hat_1.size())
        # print("phi_hat_2 : ",phi_hat_2.size())

        # self.curr_conditions['pos'][idx+1] = torch.stack([x_p+x_delta,y_p+y_delta],dim=1)
        self.curr_conditions['pos'][idx+1] = torch.where(~mask, x_1, x_2)
        mask = torch.squeeze(mask)
        self.curr_conditions['phi'][idx+1] = torch.where(~mask, phi_hat_1, phi_hat_2)
        self.curr_conditions['vel'][idx+1] = v

        # print("self.curr_conditions['pos'] : ",self.curr_conditions['pos'].size())
        # print("self.curr_conditions['phi'] : ",self.curr_conditions['phi'].size())

        return self.curr_conditions['pos'][idx+1]
        
    def reset_cur(self):
        self.curr_conditions = None

    def integrate_samples(self, v, x):
        return v

    def integrate_distribution(self, v_dist, x):
        return v_dist

# def main():
#     dynamic = Linear(0.5, None, "cuda:0", None, 0, "VEHICLE")
#     initial_dynamics = dict()
#     initial_dynamics['pos'] = torch.tensor([[1.,1.],[1.,1.]])
#     initial_dynamics['vel'] = torch.tensor([[1.,1.],[1.,1.]])
#     dynamic.set_initial_condition(initial_dynamics)
#     control_input = torch.tensor([[4.,20.],[4.,20.]])
#     for i in range(4):
#         x_t = dynamic.update_state(control_input)
#         print("x_t : ",x_t)

# main()