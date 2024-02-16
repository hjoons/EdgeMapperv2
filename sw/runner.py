import argparse

from train import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)

    parser.add_argument('--log_directory',             type=str,   help='directory to save checkpoints and summaries', default='')
    parser.add_argument('--checkpoint_path',           type=str,   help='path to a checkpoint to load', default='')
    parser.add_argument('--checkpoint_freq',           type=str,   help='checkpoint frequency', default='')
    parser.add_argument('--log_freq',                  type=int,   help='Logging frequency in global steps', default=100)

    parser.add_argument('--batch_size',                type=int,   help='batch size', default=4)
    parser.add_argument('--num_epochs',                type=int,   help='number of epochs', default=50)
    parser.add_argument('--learning_rate',             type=float, help='initial learning rate', default=1e-4)

    parser.add_argument('--do_random_rotate',                      help='if set, will perform random rotation for augmentation', action='store_true')
    parser.add_argument('--degree',                    type=float, help='random rotation maximum degree', default=2.5)

    parser.add_argument('--gt_path_eval',              type=str,   help='path to the groundtruth data for online evaluation', required=False)
    parser.add_argument('--filenames_file_eval',       type=str,   help='path to the filenames text file for online evaluation', required=False)

    parser.add_argument('--num_threads',               type=int,   help='number of threads to use for data loading', default=1)

    parser.add_argument('--seed',                      type=int,   help='random seed', default=42)
    parser.add_argument('--logname',                   type=str,   help='name of the log', default='train_logs')

    args = parser.parse_args()

    train(args)