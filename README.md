# Multitask Variational Autoencoding of Human-to-Human Object Handover

We show that it is more efficient to directly forecast the human pose and object orientation in human-to-human handover sequences as opposed to running a recurrent network when there is no need for both the intermediate poses and orientations. Published at IROS 2021.

# Contents
------------
  * [Requirements](#requirements)
  * [Brief Project Structure](#brief-project-structure)
  * [Results](#results)
  * [Training](#training)
  * [Testing](#training)
  * [Future Work](#usage)
  * [References](#references)

# Requirements
------------
What we used to develop the system

  * Ubuntu 18.04
  * PyTorch
  * PyQtGraph (for visualization)
  
# Brief Project Structure
------------

    ├── dataloader    : directory containing dataloader scripts
    ├── misc          : directory containing the python scripts for tensorboard visualization and model checkpointing
    ├── shell scripts : directory containing the shell scripts that run the train and test python scripts
    ├── models        : directory containing the model architectures
    ├── train.py      : train script
    ├── test.py       : test script
    
# Results
------------

  * Example Video of Object Handover from the [dataset webpage](https://bridges.monash.edu/articles/dataset/Handover_Orientation_and_Motion_Capture_Dataset/8287799). Note that the dataset contains only the upper body joints.


https://user-images.githubusercontent.com/16434638/125178952-7adb0f80-e1e1-11eb-94dd-76f303a7bf60.mp4


  * Given the giver and receiver input poses in transparent pink and white respectively, and the object label, we directly forecast their poses at handover shown by the brighter colors, and an appropriate orientation of the object shown by the RGB vectors. Note that the lower half of the body from the hip downwards have been manually drawn since the dataset only provides the upper half.

![output](https://user-images.githubusercontent.com/16434638/125206865-3c952d00-e281-11eb-93c7-8a929ae76fc8.png)

![output](https://user-images.githubusercontent.com/16434638/125206869-4159e100-e281-11eb-9946-3cee24877daa.png)

# Training
------------

  * `git clone https://github.com/HaziqRazali/Multitask-Variational-Autoencoding-of-Human-to-Human-Object-Handover Object-Handover`
  * Either download the [preprocessed data](https://imperialcollegelondon.box.com/s/vwvjzh2u781sui0w3ogpynq8gr4wuzwu) and put it in your `$HOME/datasets` directory or download the [raw data](https://bridges.monash.edu/articles/dataset/Handover_Orientation_and_Motion_Capture_Dataset/8287799) (60gb) and pre-process it by following the instructions at 
  * Download the GloVe vector from [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/) and put it in your `$HOME/datasets` directory. You can download any GloVe vector or even train the model with a one-hot embedding but you will have to set the appropriate `glove_type`, `path_to_glove_file` and model dimensions in the shell script e.g. `shell_scripts/train-glove-vae.sh` or `shell_scripts/train-onehot-vae.sh`.
  * cd into `shell_scripts` then run `./train-glove-vae.sh`
  * The weights will be stored in `Object-Handover/weights`
  
  
# Testing
------------
  * Set the path to your pre-trained model in `model_load_path` in `shell_scripts/test-glove-vae.sh`
  * cd into `shell_scripts` then run `./test-glove-vae.sh`
  * The output of each frame will be stored in a `.json` file in the `Object-Handover/results` directory

# Future Work
------------

How to further improve the model

  * Use synthetic data to increase number of classes and to limit the variance in object orientations.
  * Retrain GloVe on a filtered corpus without sentences containing irrelevant words e.g. ski or happy.

# References
------------

```  
@InProceedings{haziq2020handover,  
author = {Razali, Haziq and Demiris, Yiannis},  
title = {Multitask Variational Autoencoding of Human-to-Human Object Handovers},  
booktitle = {International Conference on Intelligent Robots and Systems},  
year = {2021}  
}  
```

```  
@article{chan2020affordance,
  title={An affordance and distance minimization based method for computing object orientations for robot human handovers},
  author={Chan, Wesley P and Pan, Matthew KXJ and Croft, Elizabeth A and Inaba, Masayuki},
  journal={International Journal of Social Robotics},
  volume={12},
  number={1},
  pages={143--162},
  year={2020},
  publisher={Springer}
}
```
