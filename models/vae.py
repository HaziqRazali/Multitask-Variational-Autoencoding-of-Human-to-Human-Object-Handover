import torch.distributions as tdist
import torch.nn.functional as F
import torch
import torch.nn as nn
import time

from models.components import *

class model(nn.Module):
    def __init__(self, args):
        super(model, self).__init__()
                
        for key, value in args.__dict__.items():
            setattr(self, key, value)
                                      
        self.inp_pose_encoder        = make_mlp(self.pose_encoder_units,self.pose_encoder_activations)
        self.object_label_encoder    = nn.Identity() if len(self.object_label_encoder_units) == 0 else make_mlp(self.object_label_encoder_units,self.object_label_encoder_activations)
        self.key_pose_encoder        = make_mlp(self.pose_encoder_units,self.pose_encoder_activations)
        self.object_rotation_encoder = make_mlp(self.object_rotation_encoder_units,self.object_rotation_encoder_activations)
        
        self.mu      = make_mlp(args.mu_var_units,args.mu_var_activations)
        self.log_var = make_mlp(args.mu_var_units,args.mu_var_activations)
        
        self.key_pose_decoder        = make_mlp(self.pose_decoder_units,self.pose_decoder_activations)
        self.object_rotation_decoder = make_mlp(self.object_rotation_decoder_units,self.object_rotation_decoder_activations)
                
        self.norm = tdist.Normal(torch.tensor(0.0), torch.tensor(1.0))
                        
    def forward(self, data, mode):
            
        # get data    
        object_label    = data["object_label"].view(self.batch_size,-1)        # [batch,glove dim]
        object_rotation = data["object_rotation"].view(self.batch_size,-1)     # [batch,3]
        inp_pose = data["inp_pose"].view(self.batch_size,-1) # [batch,51*2]
        key_pose = data["key_pose"].view(self.batch_size,-1) # [batch,51*2]
                
        # replicate 
        object_label    = object_label.unsqueeze(1).repeat(1, self.num_samples, 1).squeeze()
        object_rotation = object_rotation.unsqueeze(1).repeat(1, self.num_samples, 1).squeeze()
        inp_pose = inp_pose.unsqueeze(1).repeat(1, self.num_samples, 1).squeeze()
        key_pose = key_pose.unsqueeze(1).repeat(1, self.num_samples, 1).squeeze()
                           
        # feed x and y
        inp_pose_features        = self.inp_pose_encoder(inp_pose)
        object_label_features    = self.object_label_encoder(object_label)
        key_pose_features        = self.key_pose_encoder(key_pose)
        object_rotation_features = self.object_rotation_encoder(object_rotation)
        
        # compute posterior parameters
        posterior = torch.cat((inp_pose_features,object_label_features,key_pose_features,object_rotation_features),dim=-1)                    
        posterior_mu = self.mu(posterior)
        posterior_log_var = self.log_var(posterior)
        
        # sample 
        posterior_std = torch.exp(0.5*posterior_log_var)
        posterior_eps = self.norm.sample([self.batch_size, self.num_samples, posterior_mu.shape[-1]]).squeeze().cuda()
        posterior_z   = posterior_mu + posterior_eps*posterior_std
        z = posterior_z if mode == "te" else self.norm.sample([self.batch_size, self.num_samples, self.mu_var_units[-1]]).squeeze().cuda()
        
        # forecast        
        pred_key_pose = self.key_pose_decoder(torch.cat((inp_pose_features,object_label_features,z),dim=-1))
        pred_object_rotation = self.object_rotation_decoder(torch.cat((inp_pose_features,object_label_features,z),dim=-1))
                            
        return {"key_pose": pred_key_pose.view(self.batch_size,self.num_samples,-1,3).squeeze(), 
                "object_rotation": pred_object_rotation.view(self.batch_size,self.num_samples,-1).squeeze(),
                "pose_posterior":{"mu":posterior_mu, "log_var":posterior_log_var}}