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
                  
        # inp pose encoder
        self.inp_pose_encoder      = make_mlp(self.pose_encoder_units,self.pose_encoder_activations)
        
        # obj label encoder
        self.object_label_encoder  = nn.Identity() if len(self.object_label_encoder_units) == 0 else make_mlp(self.object_label_encoder_units,self.object_label_encoder_activations)
                
        # key pose decoder
        self.key_pose_decoder = make_mlp(self.pose_decoder_units,self.pose_decoder_activations)
        
        # obj rotation decoder
        self.object_rotation_decoder = make_mlp(self.object_rotation_decoder_units,self.object_rotation_decoder_activations)
                        
    def forward(self, data, mode):
            
        object_label    = data["object_label"].view(self.batch_size,-1)        # [batch,glove dim]
        object_rotation = data["object_rotation"].view(self.batch_size,-1)     # [batch,3]
        inp_pose = data["inp_pose"].view(self.batch_size,-1) # [batch,51*2]
        key_pose = data["key_pose"].view(self.batch_size,-1) # [batch,51*2]
                                                     
        # inp pose
        inp_pose = self.inp_pose_encoder(inp_pose)
        
        # obj label
        object_label = self.object_label_encoder(object_label)
                
        # pred key pose
        pred_key_pose = self.key_pose_decoder(torch.cat((inp_pose,object_label),dim=-1))
        
        # pred obj rotation
        pred_object_rotation =self.object_rotation_decoder(torch.cat((inp_pose,object_label),dim=-1))
                            
        return {"key_pose": pred_key_pose.view(self.batch_size,self.num_samples,-1,3).squeeze(), 
                "object_rotation": pred_object_rotation.view(self.batch_size,self.num_samples,-1).squeeze()}