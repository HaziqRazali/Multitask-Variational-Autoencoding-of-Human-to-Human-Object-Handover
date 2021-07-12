import os
import cv2
import math
import h5py
import json
import torch 
import random
import numpy as np
import pandas as pd
import torchvision.transforms.functional as transforms

from tqdm import tqdm
from PIL import Image
from glob import glob
from random import randint
from scipy.spatial.transform import Rotation as R

np.random.seed(1337)

train = ["0102","0201","0304","0403","0506","0605","0708","0807","0910","1009","1112","1211","1314","1413","1516","1615"]
val = ["1718","1817","1920","2019"]
test = ["1718","1817","1920","2019"]
    
class dataloader(torch.utils.data.Dataset):
    def __init__(self, args, dtype):
                
        for key, value in args.__dict__.items():
            setattr(self, key, value)
        self.dtype = dtype
        
        self.data = []
        sequence_foldernames = sorted(glob(os.path.join(args.dataset_root, "*")))
        for sequence_foldername in sequence_foldernames:
            samples = sorted(glob(os.path.join(sequence_foldername,"*"))) 
                                   
            for subject in eval(dtype):
                if subject in sequence_foldername:
                    if self.dtype != "test":
                        self.data.append(samples)
                    elif self.dtype == "test":
                        self.data.extend(samples)
                    else:
                        print(self.dtype,"error")
            
        # initialize word embeddings
        if self.path_to_glove_file:
            self.word_embeddings = {}
            with open(self.path_to_glove_file[0], encoding="utf8") as f:
                for line in f:
                    word, coefs = line.split(maxsplit=1)
                    coefs = np.fromstring(coefs, "f", sep=" ")
                    self.word_embeddings[word] = coefs
                self.word_embeddings["cereal_box"] = self.word_embeddings["cereal"]
                self.word_embeddings["dry_umbrella"] = self.word_embeddings["umbrella"]
        else:
            self.word_embeddings = one_hot_word_embeddings()
                        
        self.data_len = len(self.data)
        print("num", dtype, "samples", len(self.data))
                        
    def __len__(self):
        return max(len(self.data),self.batch_size)

    def __getitem__(self, idx):
            
            #filename_full = self.data[idx]
            #data = pd.read_csv(filename_full)
            
            idx = idx % self.data_len
            
            data = self.data[idx]        
            filename_full = random.choice(data) if type(data) == type([]) else data
            data = pd.read_csv(filename_full)
            
            # filename
            filename = filename_full.split("/")
            sequence = filename_full.split("/")[-2]
            filename = filename_full.split("/")[-1][:-4]
            
            # frame and keyframe
            frame = data["frame"].values[0]
            key_frame = data["keyframe"].values[0]
            time = np.abs(key_frame/300-frame/300)
            assert time >= 0
                            
            # get giver and receiver keypose
            giver_input_pose    = data.loc[:,data.columns.str.contains("giver_input_pose:")]
            giver_key_pose      = data.loc[:,data.columns.str.contains("giver_key_pose:")]
            receiver_input_pose = data.loc[:,data.columns.str.contains("receiver_input_pose:")]
            receiver_key_pose   = data.loc[:,data.columns.str.contains("receiver_key_pose:")]
                 
            # get object data
            object          = data.loc[:,~data.columns.str.contains("giver")&~data.columns.str.contains("receiver")&~data.columns.str.contains("frame")]
            object_label    = object.loc[0,"name"]
            object_rotation = object.loc[:, object.columns.str.contains("RX")| object.columns.str.contains("RY")| object.columns.str.contains("RZ")]
            object_position = object.loc[:,~object.columns.str.contains("RX")&~object.columns.str.contains("RY")&~object.columns.str.contains("RZ")&~object.columns.str.contains("name")]
                                            
            # process data
            # convert from milimeters to metres
            giver_input_pose    = np.reshape(giver_input_pose.values,[-1,3]) / 1000
            giver_key_pose      = np.reshape(giver_key_pose.values,[-1,3]) / 1000
            receiver_input_pose = np.reshape(receiver_input_pose.values,[-1,3]) / 1000
            receiver_key_pose   = np.reshape(receiver_key_pose.values,[-1,3]) / 1000
            object_label        = self.word_embeddings[object_label]
            object_position     = object_position.values
            object_position     = object_position[~np.isnan(object_position)] 
            object_position     = np.mean(np.reshape(object_position,[-1,3]), axis=0, keepdims=True) / 1000
            object_rotation     = np.squeeze(object_rotation.values)
                    
            # DO NOT subtract by the chest               
            giver_input_pose    = (giver_input_pose).astype(np.float32)
            giver_key_pose      = (giver_key_pose).astype(np.float32)
            receiver_input_pose = (receiver_input_pose).astype(np.float32)
            receiver_key_pose   = (receiver_key_pose).astype(np.float32)
            object_position     = (object_position).astype(np.float32)
            object_rotation     = object_rotation.astype(np.float32)
            object_label        = object_label.astype(np.float32)
            time                = time.astype(np.float32)
            
            # determine position of giver wrt to receiver
            giver_input_pose_y = np.mean(giver_input_pose[:,1])
            giver_key_pose_y   = np.mean(giver_key_pose[:,1])
            receiver_input_pose_y = np.mean(receiver_input_pose[:,1])
            receiver_key_pose_y   = np.mean(receiver_key_pose[:,1])
                    
            # rotate 180 deg
            if giver_input_pose_y < receiver_input_pose_y:
                giver_input_pose    = rotate(giver_input_pose, np.pi, "z").astype(np.float32)
                giver_key_pose      = rotate(giver_key_pose, np.pi, "z").astype(np.float32)
                receiver_input_pose = rotate(receiver_input_pose, np.pi, "z").astype(np.float32)
                receiver_key_pose   = rotate(receiver_key_pose, np.pi, "z").astype(np.float32)
            
            input_skeleton      = np.concatenate((receiver_input_pose,giver_input_pose),axis=0)
            output_skeleton     = np.concatenate((receiver_key_pose,giver_key_pose),axis=0)
                        
            object_position = np.squeeze(object_position)
                        
            return {"inp_pose":input_skeleton, "key_pose":output_skeleton, 
            "giver_input_pose":giver_input_pose, "giver_key_pose":giver_key_pose, 
            "receiver_input_pose":receiver_input_pose,  "receiver_key_pose":receiver_key_pose, 
            "object_label":object_label, "object_position":object_position, "object_rotation":object_rotation, "inp_frame":frame, "filename":filename, "sequence":sequence}

def rotate(v,theta,axis):
    
    assert axis == "x" or axis == "y" or axis == "z"
    
    if axis == "x":
        R = np.array([[1,      0,           0        ],
                      [0,np.cos(theta),-np.sin(theta)],
                      [0,np.sin(theta), np.cos(theta)]])
                      
    if axis == "y":
        R = np.array([[np.cos(theta),  0, np.sin(theta)],
                      [0,              1,      0       ],
                      [-np.sin(theta), 0, np.cos(theta)]])
                      
    if axis == "z":
        R = np.array([[np.cos(theta),  -np.sin(theta), 0],
                      [np.sin(theta),   np.cos(theta), 0],
                      [0,                    0,        1]])
                      
    return np.matmul(R,v.T).T
        
def one_hot_word_embeddings():
    
    #object_names = ["book","bottle","camera","cereal","umbrella","flowers","fork","hammer","knife","mug","pen","plate","remote","scissors","screwdriver","stapler","teapot","tomato","wineglass","wrench"]
    
    object_names = {"book":"book","bottle":"bottle","camera":"camera","cereal_box":"cereal","umbrella":"umbrella","flowers":"flowers","fork":"fork","hammer":"hammer","knife":"knife","mug":"mug","pen":"pen",
"plate":"plate","remote":"remote","scissors":"scissors","screwdriver":"screwdriver","stapler":"stapler","teapot":"teapot","tomato":"tomato","wineglass":"wineglass","wrench":"wrench"}
    
    indices = np.array([x for x in range(len(object_names.keys()))])
    one_hot = np.zeros((indices.size, indices.max()+1))
    one_hot[np.arange(indices.size),indices] = 1

    object_one_hot_dict = {}
    for i,(k,v) in enumerate(object_names.items()):
        object_one_hot_dict[k] = one_hot[i].astype(np.float32)
        
    return object_one_hot_dict