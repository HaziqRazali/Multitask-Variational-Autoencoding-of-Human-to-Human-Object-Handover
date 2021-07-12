import argparse

def argparser():

    # Argument Parser
    ##################################################### 
    parser = argparse.ArgumentParser()
    # dataset
    parser.add_argument('--dataset_root', required=True, type=str)
    parser.add_argument('--data_loader', required=True, type=str)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--path_to_glove_file', nargs="*", type=str)
    # model settings
    parser.add_argument('--architecture', required=True, type=str)
    parser.add_argument('--pose_encoder_units', nargs="*", required=True, type=int)
    parser.add_argument('--pose_encoder_activations', nargs="*", required=True, type=str)
    parser.add_argument('--object_label_encoder_units', nargs="*", type=int)
    parser.add_argument('--object_label_encoder_activations', nargs="*", type=str)
    parser.add_argument('--object_rotation_encoder_units', nargs="*", type=int)
    parser.add_argument('--object_rotation_encoder_activations', nargs="*", type=str)
    parser.add_argument('--object_position_encoder_units', nargs="*", type=str)
    parser.add_argument('--object_position_encoder_activations', nargs="*", type=str)
    parser.add_argument('--time_encoder_units', nargs="*", type=int)
    parser.add_argument('--time_encoder_activations', nargs="*", type=str)
    parser.add_argument('--mu_var_units', nargs="*", type=int)
    parser.add_argument('--mu_var_activations', nargs="*", type=str)
    parser.add_argument('--pose_mu_var_units', nargs="*", type=int)
    parser.add_argument('--pose_mu_var_activations', nargs="*", type=str)
    parser.add_argument('--pose_decoder_units', nargs="*", type=int)
    parser.add_argument('--pose_decoder_activations', nargs="*", type=str)
    parser.add_argument('--object_rotation_decoder_units', nargs="*", type=int)
    parser.add_argument('--object_rotation_decoder_activations', nargs="*", type=str)
    parser.add_argument('--object_position_decoder_units', nargs="*", type=int)
    parser.add_argument('--object_position_decoder_activations', nargs="*", type=str)
    parser.add_argument('--decode_raw', default=0, type=int)
    parser.add_argument('--sample_prior', default=0, type=int)
    # checkpointing
    parser.add_argument('--log_save_path', required=True, type=str)
    parser.add_argument('--model_save_path', required=True, type=str)
    parser.add_argument('--model_load_path', required=True, type=str)
    parser.add_argument('--restore_from_checkpoint', default=0, type=int)
    # general optimization
    parser.add_argument('--lr', required=True, type=float)
    parser.add_argument('--max_epochs', default=100000, type=int)
    parser.add_argument('--tr_step', required=True, type=int)
    parser.add_argument('--va_step', required=True, type=int)
    parser.add_argument('--loss_names', nargs="*", required=True, type=str)
    parser.add_argument('--loss_functions', nargs="*", required=True, type=str)
    parser.add_argument('--loss_weights', nargs="*", required=True, type=float)
    parser.add_argument('--debug', default=0, type=int)
    # experiments
    parser.add_argument('--sample_filename', type=str)
    parser.add_argument('--add_offsets', default=False, action='store_true')
    parser.add_argument('--num_samples', default=1, type=int)
    parser.add_argument('--eval_type', default="mean", type=str)
    # parse
    args = parser.parse_args()
    return args