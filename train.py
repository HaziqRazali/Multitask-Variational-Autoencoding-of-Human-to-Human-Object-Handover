import os
import sys
import time
import torch
import argparse
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from pathlib import Path
from collections import OrderedDict
from tensorboardX import SummaryWriter
from misc.misc import *
from misc.draw import *
from misc.args import *

from torch.nn.functional import cross_entropy, mse_loss

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
    
# Prepare Logging
##################################################### 
datetime = time.strftime("%c")  
writer = SummaryWriter(os.path.join(args.log_save_path,datetime))
checkpoint = {
    'model_summary': None,
    'best_key_pose_loss': np.inf,
    'best_key_pose_epoch': 0,
    'best_object_rotation_loss':np.inf,
    'best_object_rotation_epoch':0,
    'model_state': None,
    'optim_state': None,
    'epoch': 0,
    'tr_counter': 0,
    'va_counter': 0
}
tr_counter = 0 
va_counter = 0
epoch = 0
Path(args.model_save_path).mkdir(parents=True, exist_ok=True)
 
# Prepare Data Loaders
##################################################### 
long_dtype, float_dtype = get_dtypes(args)
# load data
#tr_data, va_data = split_tr_va(args)
tr_data = dataloader(args, "train")
va_data = dataloader(args, "val")
# data loader
tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=torch.cuda.is_available())
va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, shuffle=True, num_workers=1, drop_last=True, pin_memory=torch.cuda.is_available())

# Prepare Network and Optimizers
##################################################### 
net = model(args)
# must set model type before initializing the optimizer 
# https://discuss.pytorch.org/t/code-that-loads-sgd-fails-to-load-adam-state-to-gpu/61783/2 
print("Total # of parameters: ", count_parameters(net))
net.type(float_dtype)
optimizer  = optim.Adam(net.parameters(), lr=args.lr)
 
# Maybe load weights and initialize checkpoints
##################################################### 
restore_path = None
if args.restore_from_checkpoint == 1:
    if args.model_load_path is not None and os.path.isfile(args.model_load_path):
        # load checkpoint dictionary
        checkpoint = torch.load(args.model_load_path)
        
        print("Architecture Loading ==============")
        print(checkpoint["model_summary"])
        print("Architecture Loading ==============")
        
        # load weights from single/multi gpu
        # https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/14
        model_state = checkpoint['model_state']
        single_gpu_model_state = OrderedDict()
        for k, v in model_state.items():
            single_gpu_model_state[k.replace(".module", "")]=v
        net.load_state_dict(single_gpu_model_state)
        
        # load the rest
        optimizer.load_state_dict(checkpoint['optim_state'])
        epoch = checkpoint['epoch']+1
        tr_counter = checkpoint['tr_counter']
        va_counter = checkpoint['va_counter']
        print("Model Loaded")
    else:
        sys.exit(args.model_load_path + " not found")
        
# Main Loop
####################################################  
def loop(net, inp_data, optimizer, counter, args, mode):   
        
    # move to gpu
    for k,v in inp_data.items():
        inp_data[k] = inp_data[k].cuda() if torch.cuda.device_count() > 0 and type(v) != type([]) else inp_data[k]
           
    # Forward pass
    out_data = net(inp_data, mode=mode)
        
    # compute unscaled losses
    losses = {}
    for loss_name, loss_function in zip(args.loss_names, args.loss_functions):    
        loss = eval(loss_function)(out_data[loss_name], inp_data[loss_name], reduction="none") if loss_name in inp_data else eval(loss_function)(out_data[loss_name])
        losses[loss_name] = torch.sum(loss)  
    
    # write unscaled losses to log file
    for k,v in losses.items():
        if v != 0:
            writer.add_scalar(os.path.join(k,mode), v.item(), counter)   
    
    # Backprop the scaled losses
    total_loss = 0
    for loss_name, loss_weight in zip(args.loss_names, args.loss_weights):
        total_loss += loss_weight*losses[loss_name]
    if  mode == "tr" and optimizer is not None:
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # move all to cpu numpy
    losses = iterdict(losses)
    inp_data = iterdict(inp_data)
    out_data = iterdict(out_data)
                      
    return {"out_data":out_data, "losses":losses}

# Train
####################################################
for i in range(epoch, args.max_epochs): 
                                        
    # training ---------------------
    net.train()
    start = time.time()
    for batch_idx, tr_data in enumerate(tr_loader):
        
        #print every 100 
        if batch_idx%100 == 0:
            print("Epoch " + str(i).zfill(2) + " training batch ", batch_idx, " of ", len(tr_loader))
        
        tr_output = loop(net=net,inp_data=tr_data,optimizer=optimizer,counter=tr_counter,args=args,mode="tr")
        tr_counter= tr_counter+1
                                
        if  batch_idx!=0 and batch_idx%args.tr_step == 0:
            break
        #break
    end = time.time()
    tr_time = end - start
    # training ---------------------
    
    # validation ---------------------
    start = time.time()
    with torch.no_grad():        
        net.eval()
        va_losses = {}
        for batch_idx, va_data in enumerate(va_loader):
        
            #print every 100
            if batch_idx%50 == 0:
                print("Epoch " + str(i).zfill(2) + " validation batch ", batch_idx, " of ", len(va_loader))
            
            va_output = loop(net=net,inp_data=va_data,optimizer=None,counter=va_counter,args=args,mode="va")
            va_counter= va_counter+1 
                                    
            # accumulate loss                    
            for key in args.loss_names:
                va_losses = collect(va_losses, key=key, value=va_output["losses"][key]) if key in va_output["losses"] else va_losses
                    
            if  batch_idx!=0 and batch_idx%args.va_step == 0:
                break
                
    end = time.time()
    va_time = end - start     
    
    # average loss
    for k,v in va_losses.items():
        va_losses[k] = np.mean(np.array(va_losses[k]))
    # validation ---------------------
    
    # Save model checkpoint 
    if "object_rotation" in va_losses:
        checkpoint, save = save_model(checkpoint, net, args, optimizer, tr_counter, va_counter, i, "object_rotation", va_losses["object_rotation"])
    
    # Save model checkpoint 
    if "key_pose" in va_losses:
        loss = va_losses["key_pose"] if "pose_posterior" not in va_losses else va_losses["key_pose"] + va_losses["pose_posterior"]
        checkpoint, save = save_model(checkpoint, net, args, optimizer, tr_counter, va_counter, i, "key_pose", va_losses["key_pose"])
        if save:
            draw(tr_data, tr_output["out_data"], args, writer, i, "tr")
            draw(va_data, va_output["out_data"], args, writer, i, "va")
                        
    print("Curr Loss:       ", va_losses)
    print("Best Loss:       ", {k:v for k,v in checkpoint.items() if "loss" in k})
    print("Best Epoch:      ", {k:v for k,v in checkpoint.items() if "epoch" in k})
    print("Training time:   ", tr_time)
    print("Validation time: ", va_time)