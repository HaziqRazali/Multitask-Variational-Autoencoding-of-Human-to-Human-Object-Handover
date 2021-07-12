import os
import cv2
import torch
import numpy as np

from random import randrange
from matplotlib.figure import Figure

joints = ["forehead", "neck_base", "chest", 
"right_collarbone_midpoint", "right_shoulder", "right_upper_arm_midpoint", "right_elbow", "right_forearm_midpoint", "right_wrist", "right_hand", 
"left_collarbone_midpoint", "left_shoulder", "left_upper_arm_midpoint", "left_elbow", "left_forearm_midpoint", "left_wrist", "left_hand"]

link_names = [["forehead","neck_base"], ["neck_base", "chest"],
              ["neck_base", "left_collarbone_midpoint"], ["left_collarbone_midpoint","left_shoulder"], ["left_shoulder", "left_upper_arm_midpoint"], 
              ["left_upper_arm_midpoint","left_elbow"], ["left_elbow","left_forearm_midpoint"], ["left_forearm_midpoint","left_wrist"], ["left_wrist", "left_hand"],
              ["neck_base", "right_collarbone_midpoint"], ["right_collarbone_midpoint","right_shoulder"], ["right_shoulder", "right_upper_arm_midpoint"], 
              ["right_upper_arm_midpoint","right_elbow"], ["right_elbow","right_forearm_midpoint"], ["right_forearm_midpoint","right_wrist"], ["right_wrist", "right_hand"]]
link_ids = [[joints.index(a),joints.index(b)] for a,b in link_names]

def draw(net_inp, net_out, args, writer, epoch, mode):
       
    batch=randrange(len(net_inp["key_pose"]))  
    fig = Figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
                
    # https://matplotlib.org/3.1.1/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html
    ax.text2D(0.05, 0.95, "Ground Truth", fontsize=30, color='red',  transform=ax.transAxes)
    ax.text2D(0.05, 0.90, "Prediction",   fontsize=30, color='blue', transform=ax.transAxes)
    ax.text2D(0.05, 0.85, str(net_inp["object_rotation"][batch]), fontsize=30, color='red',  transform=ax.transAxes)
    ax.text2D(0.05, 0.80, str(net_out["object_rotation"][batch]), fontsize=30, color='blue', transform=ax.transAxes)
    
    filename = net_inp["filename"][batch]
    ax.text2D(0.05, 0.75, filename, fontsize=20, transform=ax.transAxes)
    
    # # # # # # # # # # # # #
    # draw object rotation  # 
    # # # # # # # # # # # # #
                
    object_position = net_inp["giver_key_pose"][batch][9]
    ax.scatter(object_position[0],object_position[1],object_position[2],s=40,c="red")
        
    # ground truth rotation
    thetas = net_inp["object_rotation"][batch]
    xyz_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])*0.1
    for theta,axis in zip(thetas,["x","y","z"]):
        xyz_vectors = rotate(xyz_vectors,theta,axis)
    for xyz_vector in xyz_vectors:
        ax.plot([object_position[0],object_position[0]+xyz_vector[0]],
                [object_position[1],object_position[1]+xyz_vector[1]],
                [object_position[2],object_position[2]+xyz_vector[2]],color="red")
                
    # predicted rotation
    thetas = net_out["object_rotation"][batch]
    xyz_vectors = np.array([[1,0,0],[0,1,0],[0,0,1]])*0.1
    for theta,axis in zip(thetas,["x","y","z"]):
        xyz_vectors = rotate(xyz_vectors,theta,axis)
    for xyz_vector in xyz_vectors:
        ax.plot([object_position[0],object_position[0]+xyz_vector[0]],
                [object_position[1],object_position[1]+xyz_vector[1]],
                [object_position[2],object_position[2]+xyz_vector[2]],color="blue")
             
    # # # # # # # # # 
    # draw poses # 
    # # # # # # # # #  
       
    receiver_key_pose = net_out["key_pose"][batch,:17]
    giver_key_pose    = net_out["key_pose"][batch,17:]
       
    # predicted receiver key pose
    for a,b in link_ids:
        ax.plot([receiver_key_pose[a,0],receiver_key_pose[b,0]],
                [receiver_key_pose[a,1],receiver_key_pose[b,1]],
                [receiver_key_pose[a,2],receiver_key_pose[b,2]],color='red')         
    # predicted giver key pose
    for a,b in link_ids:
        ax.plot([giver_key_pose[a,0],giver_key_pose[b,0]],
                [giver_key_pose[a,1],giver_key_pose[b,1]],
                [giver_key_pose[a,2],giver_key_pose[b,2]],color='blue')      
                  
    writer.add_figure(mode+"_fig",fig,epoch)
    
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
                      
    return np.matmul(R,v)