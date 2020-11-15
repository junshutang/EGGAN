import argparse

def get_config():
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
    parser.add_argument('--au_dim', type=int, default=17, help='number of aus')
    parser.add_argument('--id_classes', type=int, default=27, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=128, help='size of each image dimension')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--paral', type=bool, default=False)

    # Training configuration
    parser.add_argument('--n_epochs', type=int, default=4, help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

    parser.add_argument('--save_freq', type=int, default=1, help='epoch between model saving')
    parser.add_argument('--train_shuffle', type=str, default=False)
    parser.add_argument('--version', type=str, default='v1')

    parser.add_argument('--lambda_au', type=int, default=100)
    parser.add_argument('--lambda_gp', type=int, default=10)
    parser.add_argument('--lambda_rec', type=int, default=100)
    parser.add_argument('--lambda_id', type=int, default=60)
    parser.add_argument('--lambda_pe', type=int, default=20)
    parser.add_argument('--lambda_ms', type=int, default=1)

    # Testing configuration
    parser.add_argument('--test_exc_path', type=str, default='test_exc')
    parser.add_argument('--test_interp_path', type=str, default='test_interp')
    parser.add_argument('--test_epoch', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default=1)
    parser.add_argument('--test_shuffle', type=bool, default=False)
    parser.add_argument('--test_src_num', type=int, default=0)
    parser.add_argument('--test_tgt_num', type=int, default=1)


    # Directories
    parser.add_argument('--save_dir', type=str, default='checkpoint/')
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--attr_dir', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default=None)

    config = parser.parse_args()

    return config
