import os
import cv2
import sys
import ast
import time
import torch
import numpy as np
import argparse
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from misc.args import *
from misc.misc import *
from misc.draw import *

torch.manual_seed(1337)

# Argument Parser
##################################################### 
args = argparser()

# Imports for Architecture and Data Loader 
##################################################### 
architecture_import = "from models.{} import *".format(args.architecture)
exec(architecture_import)
data_loader_import = "from dataloaders.{} import *".format(args.data_loader)
exec(data_loader_import)
         
# Prepare Data Loaders
##################################################### 
long_dtype, float_dtype = get_dtypes(args)
# load data
va_data = dataloader(args, "test")
# data loader
va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=True, pin_memory=torch.cuda.is_available())
    
# Prepare Network and Optimizers
##################################################### 
net = model(args)
# must set model type before initializing the optimizer 
# https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/2 
print("Total # of parameters: ", count_parameters(net))
net.type(float_dtype)

# Load weights and initialize checkpoints
##################################################### 
print("Attempting to load from: " + args.model_load_path)
print("Locatable? : " + str(os.path.isfile(args.model_load_path)))
if args.model_load_path is not None and os.path.isfile(args.model_load_path):

    # load checkpoint dictionary
    checkpoint = torch.load(args.model_load_path)
        
    model_state = checkpoint['model_state']
    net.load_state_dict(model_state)
    
    print("Model Loaded")
else:
    sys.exit(args.model_load_path + " not found")  

# Save Function
#################################################### 
joints_xyz = [[j+"_x",j+"_y",j+"_z"] for j in joints]
joints_xyz = [y for x in joints_xyz for y in x]
giver_pose_header = ["predicted_giver_key_pose:"+x for x in joints_xyz]
receiver_pose_header = ["predicted_receiver_key_pose:"+x for x in joints_xyz]
rotation_header = ["predicted_RX", "predicted_RY", "predicted_RZ"]
prediction_header = receiver_pose_header+giver_pose_header+rotation_header
time_header = ["predicted_time"]
def save_results(va_data, va_output):
    
    #print(va_output["giver_key_pose"].shape)  # (64, 256, 17, 3)
    #print(va_output["object_rotation"].shape) # (64, 256, 3)
    #print(va_output["time"].shape)            # (64, 256, 1)
    
    if args.num_samples > 1:
        va_output["output_skeleton"] = va_output["output_skeleton"][:,0,:,:] # (64, 17, 3)
        va_output["object_rotation"] = va_output["object_rotation"][:,0,:] # (64, 3)
        #va_output["time"] = None if va_output["time"] is None else va_output["time"][:,0,:] # (64, 1)
            
    for i in range(args.batch_size):
                
        # read the ground truth dataframe
        df = pd.read_csv(va_data["filename_full"][i])
                
        # form the prediction dataframe
        predicted_giver_key_pose  = np.expand_dims(va_output["output_skeleton"][i].flatten(),axis=0) # [1, 51]
        predicted_object_rotation = np.expand_dims(va_output["object_rotation"][i],axis=0)           # [1, 3]
        predicted_df = pd.DataFrame(np.concatenate((predicted_giver_key_pose,predicted_object_rotation),axis=1), columns=prediction_header)
        
        #if va_output["time"] is not None:
        #    predicted_time = va_output["time"][i]
        #    predicted_time = np.expand_dims(predicted_time,axis=0) # [1, 1]
        #    predicted_df = pd.DataFrame(np.concatenate((predicted_giver_key_pose,predicted_object_rotation,predicted_time),axis=1), columns=prediction_header+time_header)
        
        # concatenate them and save
        save_df = pd.concat([df,predicted_df],axis=1)
        print(os.path.join(args.save_results_path, va_data["filename"][i]))
        save_df.to_csv(os.path.join(args.save_results_path, va_data["filename"][i]),index=False)
    
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, args, mode):   
 
    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and type(v) != type([]) else inp_data[k]
           
    # Forward pass
    out_data = net(inp_data, mode=mode)
        
    if args.num_samples > 1:
        pose_loss_scalar = torch.mean(pose_loss_scalar,dim=1)
        rotation_loss_scalar = torch.mean(rotation_loss_scalar,dim=1)
                        
    # move all to cpu numpy
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
                          
    return {"out_data":out_data}

# save results 
####################################################  
def save(out, inp, args):
        
    # handle conflicting names
    keys = set(inp.keys()) | set(out.keys())
    for key in keys:
        if key in inp and key in out:
            inp["_".join(["true",key])] = inp[key]
            out["_".join(["pred",key])] = out[key]
            del inp[key]
            del out[key]
    
    # merge dict
    data = {**inp, **out}
    
    # remove components i dont need
    data = {k:v for k,v in data.items() if "posterior" not in k}
    
    # json can only save list
    for k,v in data.items():
        data[k] = data[k].tolist() if isinstance(v, type(np.array([0]))) else data[k]
                
    for i in range(len(data["sequence"])):
                
        # create folder
        model = args.model_load_path.split("/")[-2]        
        foldername = os.path.join("results",model,data["sequence"][i])
        path = Path(foldername)
        path.mkdir(parents=True,exist_ok=True)
        
        # saving file
        filename = os.path.join("results",model,data["sequence"][i],str(int(data["inp_frame"][i])).zfill(10)+".json")
        data_i = {k:v[i] for k,v in data.items()}
        print("Saving",filename)
        with open(filename, 'w') as f:
            json.dump(data_i, f)
   
# validation ---------------------
with torch.no_grad():        
    net.eval()
    va_losses = {}
    for batch_idx, va_data in enumerate(va_loader):
    
        print("Validation batch ", batch_idx, " of ", len(va_loader))
        
        # forward pass
        va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=None,args=args,mode="va")
        
        # save results
        save(va_output["out_data"], va_data, args)