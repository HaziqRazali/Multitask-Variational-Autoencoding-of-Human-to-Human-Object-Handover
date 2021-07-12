import os
import torch
import numpy as np
import torch.nn.functional as F
from collections import MutableMapping

# Save function
####################################################  
def save_model(checkpoint, net, args, optimizer, tr_counter, va_counter, best_epoch, task_name, best_loss):
    
    if best_loss > checkpoint["best_"+task_name+"_loss"]:
        return checkpoint, 0
        
    print("Saving", task_name)

    checkpoint['model_summary'] = str(net)
    checkpoint['args']          = str(args)
    checkpoint['best_'+task_name+'_loss']  = best_loss
    checkpoint['best_'+task_name+'_epoch'] = best_epoch
    checkpoint['model_state']   = net.state_dict()
    checkpoint['optim_state']   = optimizer.state_dict() 
    checkpoint['epoch']         = best_epoch
    checkpoint['tr_counter']    = tr_counter
    checkpoint['va_counter']    = va_counter   

    #if  args.debug==False:
    checkpoint_path = os.path.join(args.model_save_path, task_name+"_epoch"+str(best_epoch).zfill(4)+"best"+str(checkpoint['best_'+task_name+'_epoch']).zfill(4)+".pt")
    torch.save(checkpoint, checkpoint_path)
        
    return checkpoint, 1

# Convert dictionary data type
####################################################  
def iterdict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            iterdict(v)
        else:
            if not isinstance(v, list):
                v = v.cpu().detach().numpy()
            d.update({k: v})
    return d
    
# Flatten dictionary
####################################################  
def flattendict(d, parent_key ='', sep ='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
  
        if isinstance(v, MutableMapping):
            items.extend(flattendict(v, new_key, sep = sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
    
# Set data type
####################################################  
def get_dtypes(args):
    long_dtype = torch.LongTensor
    float_dtype = torch.FloatTensor
    if torch.cuda.device_count() > 0:
        long_dtype = torch.cuda.LongTensor
        float_dtype = torch.cuda.FloatTensor
    return long_dtype, float_dtype

# Accumulate list in dictionary
####################################################          
def collect(dictionary, key, value):
    if key not in dictionary:
        dictionary[key] = [value]
    else:
        dictionary[key].append(value)
    return dictionary
    
# count the total number of parameters
####################################################    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
# KL Divergence
####################################################    
def kl_loss(out_data):
    # inp_data not used
    return -0.5 * torch.sum(out_data["log_var"] - out_data["log_var"].exp() - out_data["mu"].pow(2) +1 )