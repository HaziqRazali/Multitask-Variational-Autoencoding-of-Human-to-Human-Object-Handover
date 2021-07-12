source "/opt/anaconda3/etc/profile.d/conda.sh"
conda activate ~/cenvs/hpf

# optimization
gpu_num="0"
lr=1e-3
tr_step=1
va_step=1
loss_names="key_pose object_rotation pose_posterior"
loss_functions="mse_loss mse_loss kl_loss"
loss_weights="0.8 0.2 0.8"

# dataset
dataset_root="$HOME/datasets/handover/data/"
data_loader="handover"
batch_size=64
glove_type="glove.6B.50d.txt"
path_to_glove_file="$HOME/datasets/handover/glove/"$glove_type

# model
# Check and make sure that the units are correct wrt the model
architecture="vae"
pose_encoder_units="102 128"
pose_encoder_activations="relu"
object_label_encoder_units=""
object_label_encoder_activations=""
pose_decoder_units="194 256 102" # pose + object_label + mu_var
pose_decoder_activations="relu none"
object_rotation_decoder_units="194 256 3"
object_rotation_decoder_activations="relu none"
object_rotation_encoder_units="3 16"
object_rotation_encoder_activations="relu"
mu_var_units="322 128 16" # pose*2 + object_label + object_rotation
mu_var_activations="relu none"

num_samples=1

cd ..
CUDA_VISIBLE_DEVICES=${gpu_num} python test.py \
--lr $lr --tr_step $tr_step --va_step $va_step --loss_names $loss_names --loss_functions $loss_functions --loss_weights $loss_weights \
--dataset_root $dataset_root --data_loader $data_loader --batch_size $batch_size --path_to_glove_file $path_to_glove_file \
--architecture $architecture \
--pose_encoder_units $pose_encoder_units --pose_encoder_activations $pose_encoder_activations \
--object_label_encoder_units $object_label_encoder_units --object_label_encoder_activations $object_label_encoder_activations \
--object_rotation_encoder_units $object_rotation_encoder_units --object_rotation_encoder_activations $object_rotation_encoder_activations \
--pose_decoder_units $pose_decoder_units --pose_decoder_activations $pose_decoder_activations \
--object_rotation_decoder_units $object_rotation_decoder_units --object_rotation_decoder_activations $object_rotation_decoder_activations \
--mu_var_units $mu_var_units --mu_var_activations $mu_var_activations \
--num_samples $num_samples \
--log_save_path "" \
--model_save_path "" \
--model_load_path "$HOME/Object-Handover/weights/(1e-3)(0.8 0.2 0.8)(64)(glove.6B.50d.txt)(vae)(102 128)()(3 16)(322 128 16)(194 256 102)(194 256 3)/key_pose_epoch0000best0000.pt"
cd shell_scripts