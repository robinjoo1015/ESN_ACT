import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

import torch
from torchvision import models
import pickle
import numpy as np
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge, ESN


from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

# resnet18
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

class FrozenBatchNorm2d(torch.nn.Module):
    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

class BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs

class Backbone(BackboneBase):
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True,
            norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

def build_backbone(args):
    train_backbone = args['lr_backbone'] > 0
    return_interm_layers = args['masks']
    backbone = Backbone(args['backbone'], train_backbone, return_interm_layers, args['dilation'])
    return backbone


class TorchReservoirModel:
    def __init__(self, 
                 n_reservoir=None, 
                 lr=None, 
                 sr=None, 
                 input_scaling=None, 
                 input_connectivity=None, 
                 rc_connectivity=None,
                 activation="tanh", 
                 input_dim=1, 
                 ridge=0.0, 
                 data_info=None,
                 device=None, load=None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)

        if load is not None:
            import gzip
            with gzip.open(load, 'rb') as f:
                loaded_tensors = torch.load(f)

            self.W = loaded_tensors['W'].to(self.device)
            self.W_in = loaded_tensors['W_in'].to(self.device)
            self.W_out = loaded_tensors['W_out'].to(self.device)
            self.res_bias = loaded_tensors['res_bias'].to(self.device)
            self.out_bias = loaded_tensors['out_bias'].to(self.device)

            self.n_reservoir = loaded_tensors['n_reservoir']
            self.lr = loaded_tensors['lr']
            self.sr = loaded_tensors['sr']
            self.input_scaling = loaded_tensors['input_scaling']
            self.input_connectivity = loaded_tensors['input_connectivity']
            self.rc_connectivity = loaded_tensors['rc_connectivity']
            self.input_dim = loaded_tensors['input_dim']
            self.ridge = loaded_tensors['ridge']

            self.data_info = loaded_tensors['data_info']

            self.activation = self._sigmoid
            if loaded_tensors['activation'] == 'tanh':
                self.activation = self._tanh

        else:
            self.n_reservoir = n_reservoir
            self.lr = lr
            self.input_dim = input_dim
            self.ridge = ridge
            self.sr = sr
            self.input_scaling = input_scaling
            self.input_connectivity = input_connectivity
            self.rc_connectivity = rc_connectivity
            self.data_info = data_info
    
            if activation.lower() == "tanh":
                self.activation = self._tanh
            elif activation.lower() == "sigmoid":
                self.activation = self._sigmoid
            else:
                raise ValueError("Unsupported activation: choose 'tanh' or 'sigmoid'")

            ##### input int #####
            if type(input_dim) is int:
                if type(input_scaling) is not float or type(input_connectivity) is not float:
                    raise ValueError("Input int mismatch")
                
                temp_reservoir = Reservoir(n_reservoir, lr=lr, sr=sr,
                                       input_scaling=input_scaling,
                                       input_connectivity=input_connectivity,
                                       rc_connectivity=rc_connectivity,
                                       input_dim=input_dim)
                temp_reservoir.initialize()
                
                self.W = self._convert_to_torch(temp_reservoir.W)
                self.W_in = self._convert_to_torch(temp_reservoir.Win)
                if hasattr(temp_reservoir, 'bias') and temp_reservoir.bias is not None:
                    self.res_bias = self._convert_to_torch(temp_reservoir.bias.toarray().reshape(-1))
                else:
                    self.res_bias = torch.zeros((n_reservoir,), dtype=torch.float64, device=self.device)
        
                del temp_reservoir

            ##### input list #####
            elif type(input_dim) is list:
                if len(input_dim) != len(input_scaling) or len(input_dim) != len(input_connectivity):
                    raise ValueError("Input list mismatch")

                W_in_list = []
                # W_temp = None
                for input_index in range(len(input_dim)):
                    if input_index == 0:
                        temp_reservoir = Reservoir(n_reservoir, lr=lr, sr=sr,
                                               input_scaling=input_scaling[input_index],
                                               input_connectivity=input_connectivity[input_index],
                                               rc_connectivity=rc_connectivity,
                                               input_dim=input_dim[input_index])
                        temp_reservoir.initialize()

                        self.W = self._convert_to_torch(temp_reservoir.W)
                        if hasattr(temp_reservoir, 'bias') and temp_reservoir.bias is not None:
                            self.res_bias = self._convert_to_torch(temp_reservoir.bias.toarray().reshape(-1))
                        else:
                            self.res_bias = torch.zeros((n_reservoir,), dtype=torch.float64, device=self.device)
                        
                        W_in_list.append(self._convert_to_torch(temp_reservoir.Win))
                    else:
                        temp_reservoir_2 = Reservoir(n_reservoir, 
                                                     # lr=lr, sr=sr,
                                                     input_scaling=input_scaling[input_index],
                                                     input_connectivity=input_connectivity[input_index],
                                                     # rc_connectivity=rc_connectivity,
                                                     input_dim=input_dim[input_index],
                                                     W=temp_reservoir.W,
                                                     bias=temp_reservoir.bias,
                                                  )
                        temp_reservoir_2.initialize()
                        W_in_list.append(self._convert_to_torch(temp_reservoir_2.Win))
                        del temp_reservoir_2

                self.W_in = torch.cat(W_in_list, dim=1)
                del temp_reservoir
    
            self.W_out = None
            self.out_bias = None
    
        self.state = torch.zeros((self.n_reservoir,), dtype=torch.float64, device=self.device)
        self.state_out = None

    def _convert_to_torch(self, mat):
        if sp.issparse(mat):
            return torch.tensor(mat.toarray(), dtype=torch.float64, device=self.device)
        elif isinstance(mat, np.ndarray):
            return torch.tensor(mat, dtype=torch.float64, device=self.device)
        elif isinstance(mat, torch.Tensor):
            return mat.to(self.device)
        else:
            raise ValueError("Unsupported type for conversion to torch tensor.")
    
    def _tanh(self, x):
        return torch.tanh(x)
    
    def _sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def step(self, u_t):
        # Ensure u_t is on the correct device.
        u_t = u_t.to(self.device)
        pre_activation = self.W_in @ u_t + self.W @ self.state + self.res_bias
        activated = self.activation(pre_activation)
        new_state = (1 - self.lr) * self.state + self.lr * activated
        self.state = new_state
        
        if self.W_out is not None:
            return self.out_bias + (self.state @ self.W_out)
        else:
            return None
    
    def train(self, X, Y, ridge=None, input_bias=True):
        if ridge is None:
            ridge = self.ridge
        self.W_out = None
        self.out_bias = None

        Y = torch.tensor(Y, dtype=torch.float64, device=self.device)

        B, T, _ = X.shape
        D_out = Y.shape[2]
        N = B * T

        X_acc = torch.zeros((N, self.n_reservoir), dtype=torch.float64, device=self.device)
        Y_acc = Y.reshape((N, D_out)) # 250416

        idx = 0
        for b in range(B):
            self.state = torch.zeros((self.n_reservoir,), dtype=torch.float64, device=self.device)
            X_b = torch.tensor(X[b, :, :], dtype=torch.float64, device=self.device) # 250416
            for t in range(T):
                u_t = X_b[t, :] # 250416
                self.step(u_t)
                X_acc[idx, :] = self.state.clone()
                idx += 1

        del X, Y; torch.cuda.empty_cache()

        if input_bias:
            ones = torch.ones((N, 1), dtype=torch.float64, device=self.device)
            X_aug = torch.cat([ones, X_acc.clone()], dim=1)
        else:
            X_aug = X_acc.clone()

        del X_acc; torch.cuda.empty_cache()

        m = X_aug.shape[1]
        W_hat = (torch.linalg.inv(X_aug.t() @ X_aug + ridge * torch.eye(m, dtype=torch.float64, device=self.device)) @ (X_aug.t() @ Y_acc))

        if input_bias:
            self.out_bias = W_hat[0, :].clone()
            self.W_out = W_hat[1:, :].clone()
        else:
            self.out_bias = torch.zeros((D_out,), dtype=torch.float64, device=self.device)
            self.W_out = W_hat.clone()
        
        del W_hat; torch.cuda.empty_cache()

    def predict(self, X, input_bias=True):
        X = torch.tensor(X, dtype=torch.float64, device=self.device)
        B, T, _ = X.shape
        D_out = self.out_bias.shape[0] if self.out_bias is not None else 0
        predictions = torch.zeros((B, T, D_out), dtype=torch.float64, device=self.device)

        for b in range(B):
            self.state = torch.zeros((self.n_reservoir,), dtype=torch.float64, device=self.device)
            for t in range(T):
                u_t = X[b, t, :]
                y_t = self.step(u_t)
                if y_t is None:
                    predictions[b, t, :] = torch.zeros((D_out,), dtype=torch.float64, device=self.device)
                else:
                    predictions[b, t, :] = y_t.clone()
        
        del X; torch.cuda.empty_cache()
        return predictions

    def save_model(self, directory='./reservoir_dict.pth.gz'):
        import gzip

        activation = 'sigmoid'
        if self.activation == self._tanh:
            activation = 'tanh'
        reservoir_dict = {
            'W': self.W.cpu(),
            'W_in': self.W_in.cpu(),
            'W_out': self.W_out.cpu(),
            'res_bias': self.res_bias.cpu(),
            'out_bias': self.out_bias.cpu(),

            'n_reservoir': self.n_reservoir,
            'lr': self.lr,
            'sr': self.sr,
            'input_scaling': self.input_scaling,
            'input_connectivity': self.input_connectivity,
            'rc_connectivity': self.rc_connectivity,
            'activation': activation,
            'input_dim': self.input_dim,
            'ridge': self.ridge,

            'data_info': self.data_info,
        }
        
        with gzip.open(directory, 'wb') as f:
            torch.save(reservoir_dict, f)

    def reset_state(self):
        self.state = torch.zeros((self.n_reservoir,), dtype=torch.float64, device=self.device)


class ESNPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        self.device = torch.device('cuda')
        # # mobilenet
        # mobilenet = models.mobilenet_v3_small(pretrained=True)
        # mobilenet = torch.nn.Sequential(*list(mobilenet.children())[:-1])
        # mobilenet = torch.nn.Sequential(*list(torch.nn.Sequential(*list(mobilenet.children())[0]).children())[:-1])
        # mobilenet.to(self.device)
        # mobilenet.eval()
        # self.backbone = mobilenet
        # self.x_min = -18.84234619140625
        # self.x_max = 19.60402488708496
        # self.y_min = -1.5885518789291382
        # self.y_max = 1.5826531648635864
        
        # resnet
        backbone = build_backbone({'lr_backbone': 0, 'backbone' : 'resnet18', 'masks': False, 'dilation': False})
        backbone.to(self.device)
        backbone.eval()
        self.backbone = backbone

        # # transfer
        # self.x_min = 0.0
        # self.x_max = 25.788471221923828
        # self.y_min = -1.5885518789291382
        # self.y_max = 1.5826531648635864
        # # insertion
        # self.x_min = 0.0
        # # self.x_max = 28.740188598632812 #
        # self.x_max = 28.720340728759766
        # self.y_min = -1.604706048965454
        # self.y_max = 1.2544714212417603

        # insertion
        # self.model = TorchReservoirModel(load='/data/ysjoo/home/projects/esn/checkpoint/torch_reservoir/resnet/250520/sim_insertion_scripted/n10000_opt10_train10/250519_insertion_n10000_opt10_train10.pth.gz')
        # transfer
        # self.model = TorchReservoirModel(load='/data/ysjoo/home/projects/esn/checkpoint/torch_reservoir/resnet/250520/sim_transfer_cube_scripted/n5000/250519_transfer_n5000_std_best.pth.gz')
        # multi
        self.model = TorchReservoirModel(load='/data/ysjoo/home/projects/esn/checkpoint/torch_reservoir/resnet/250520/sim_multi/n10000_opt10_train40/250519_multi_n10000_opt10_train40.pth.gz')

        # 250508 standardization
        # self.x_mean = self.model.data_info['x_mean']
        # self.x_std = self.model.data_info['x_std']
        self.y_mean = self.model.data_info['y_mean']
        self.y_std = self.model.data_info['y_std']
        # 250508 normalization
        self.x_min = self.model.data_info['x_min']
        self.x_max = self.model.data_info['x_max']
        # self.y_min = self.model.data_info['y_min']
        # self.y_max = self.model.data_info['y_max']

        # self.x_mean = torch.from_numpy(self.x_mean).to(self.device, dtype=torch.float64)
        # self.x_std = torch.from_numpy(self.x_std).to(self.device, dtype=torch.float64)
        # self.y_mean = torch.from_numpy(self.y_mean).to(self.device, dtype=torch.float64)
        # self.y_std = torch.from_numpy(self.y_std).to(self.device, dtype=torch.float64)


    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image).to(self.device)
        if actions is not None: # training time
            pass
        else:
            with torch.no_grad():
                features = self.backbone(image)
            # # mobilenet
            # # features = features.numpy()
            # y_curr = (torch.from_numpy(qpos).to(self.device, dtype=torch.float64) - self.y_min) / (self.y_max - self.y_min)
            # features = (features.to(dtype=torch.float64) - self.x_min) / (self.x_max - self.x_min)
            # resnet
            # features = features.numpy()

            # minmax normalization
            # y_curr = (torch.from_numpy(qpos).to(self.device, dtype=torch.float64) - self.y_min) / (self.y_max - self.y_min)
            features = (features['0'].to(dtype=torch.float64) - self.x_min) / (self.x_max - self.x_min)
            # standardization
            # y_curr = torch.from_numpy((qpos - self.y_mean) / self.y_std).to(self.device, dtype=torch.float64)
            # features = (features['0'].to(dtype=torch.float64) - self.x_mean) / self.x_std
            # std + norm
            # std_qpos = (qpos - self.y_mean) / self.y_std
            # y_curr = torch.from_numpy((std_qpos - self.y_min) / (self.y_max - self.y_min)).to(self.device, dtype=torch.float64)
            # std_feature = (features['0'].to(dtype=torch.float64) - self.x_mean) / self.x_std
            # features = (std_feature - self.x_min) / (self.x_max - self.x_min)

            # # mobilenet
            # # input_feature = np.concatenate([features.reshape((28800)), y_curr.reshape((14))])
            # input_feature = torch.cat([features.reshape((28800)), y_curr.reshape((14))])
            # resnet
            # input_feature = torch.cat([features.reshape((153600)), y_curr.reshape((14))])
            # no joint input
            input_feature = features

            step = self.model.step(u_t=input_feature.reshape((-1)))

            # minmax normalization
            # a_hat = step.detach().cpu().numpy().reshape((-1)) * (self.y_max - self.y_min) + self.y_min
            # standardization
            a_hat = step.detach().cpu().numpy().reshape((-1)).reshape((50,14)) * self.y_std + self.y_mean
            a_hat = a_hat.reshape((-1,))
            # std + norm
            # a_hat_norm = step.detach().cpu().numpy().reshape((-1)).reshape((50,14)) * (self.y_max - self.y_min) + self.y_min
            # a_hat = a_hat_norm * self.y_std + self.y_mean
            # a_hat = a_hat.reshape((-1,))

            # a_hat = readout_2.reshape((140)) * (self.y_max - self.y_min) + self.y_min
            # a_hat = readout_1.reshape((14)) * (self.y_max - self.y_min) + self.y_min

            return a_hat
    
    def reset(self):
        # self.reservoir = self.reservoir.reset()
        # self.reservoir_2 = self.reservoir_2.reset()
        self.model.reset_state()




# class ESNPolicy(nn.Module):
#     def __init__(self, args_override):
#         super().__init__()
#         # # efficientnet
#         # efficientnet = models.efficientnet_b0(pretrained=True)
#         # efficientnet.eval()
#         # efficientnet = torch.nn.Sequential(*list(efficientnet.children())[:-1])
#         # self.backbone = efficientnet
#         # resnet18
#         # backbone = build_backbone({'lr_backbone': 0, 'backbone' : 'resnet18', 'masks': False, 'dilation': False})
#         # backbone.eval()
#         # self.backbone = backbone
#         # mobilenet
#         mobilenet = models.mobilenet_v3_small(pretrained=True)
#         mobilenet = torch.nn.Sequential(*list(mobilenet.children())[:-1])
#         mobilenet = torch.nn.Sequential(*list(torch.nn.Sequential(*list(mobilenet.children())[0]).children())[:-1])
#         mobilenet.eval()
#         self.backbone = mobilenet

#         # # efficientnet
#         # with open('/home/ysjoo/projects/esn/checkpoint/sim_transfer_cube_scripted/esn_trained_model_chunk10.pickle', 'rb') as f:
#         #     model = pickle.load(f)
#         # resnet18
#         # with open('/home/ysjoo/projects/esn/checkpoint/sim_transfer_cube_scripted/resnet18/esn_trained_model_resnet18.pickle', 'rb') as f:
#         #     model = pickle.load(f)
#         # mobilenet
#         with open('/home/ysjoo/projects/esn/checkpoint/sim_transfer_cube_scripted/mobilenet/chunk10/esn_trained_model_mobilenet_chunk10.pickle', 'rb') as f:
#             model = pickle.load(f)
#         self.reservoir, self.readout, self.reservoir_2, self.readout_2, (self.x_min, self.x_max, self.y_min, self.y_max) = model

#     def __call__(self, qpos, image, actions=None, is_pad=None):
#         env_state = None
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                          std=[0.229, 0.224, 0.225])
#         image = normalize(image)
#         if actions is not None: # training time
#             pass
#         else:
#             with torch.no_grad():
#                 features = self.backbone(image)
#             # # efficientnet
#             # features = features.numpy()
#             # y_curr = (qpos - self.y_min) / (self.y_max - self.y_min)
#             # resnet18
#             # features = features['0'].numpy()
#             # y_curr = (qpos - self.y_min) / (self.y_max - self.y_min)
#             # mobilenet
#             features = features.numpy()
#             y_curr = (qpos - self.y_min) / (self.y_max - self.y_min)
#             features = (features - self.x_min) / (self.x_max - self.x_min)

#             # # efficientnet
#             # input_feature = np.concatenate([features.reshape((1280)), y_curr.reshape((14))])
#             # state_1 = self.reservoir(input_feature)
#             # resnet18
#             # input_feature = np.concatenate([features.reshape((153600)), y_curr.reshape((14))])
#             # state_1 = self.reservoir(input_feature)
#             # mobilenet
#             input_feature = np.concatenate([features.reshape((28800)), y_curr.reshape((14))])
#             state_1 = self.reservoir(input_feature)

#             readout_1 = self.readout(state_1)
#             input_feature_2 = np.concatenate([readout_1.reshape((140)), y_curr.reshape((14))])
#             state_2 = self.reservoir_2(input_feature_2)
#             readout_2 = self.readout_2(state_2)

#             a_hat = readout_2.reshape((140)) * (self.y_max - self.y_min) + self.y_min

#             # a_hat = readout_1.reshape((14)) * (self.y_max - self.y_min) + self.y_min

#             return a_hat
    
#     def reset(self):
#         self.reservoir = self.reservoir.reset()
#         self.reservoir_2 = self.reservoir_2.reset()



class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
