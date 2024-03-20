import argparse
from train import train
from eval import eval

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_path',         help='Training dataset path', type=str, required=True)
    parser.add_argument('--test_path',          help='Testing dataset path', type=str, required=True)
    
    parser.add_argument('--eval',               help='Set to enable eval mode only', action='store_true')
    parser.add_argument('--batch_size',         help='Batch size for training', type=int, default=60)
    parser.add_argument('--epochs',             help='Epochs for training and testing', type=int, default=100)
    parser.add_argument('--lr',                 help='Optimizer learning rate', type=float, default=1e-4)
    
    parser.add_argument('--ckpt_path',          help='Specify a checkpoint to use', type=str, default='')
    
    parser.add_argument('--ckpt_freq',          help='Checkpoint frequency', type=int, default=25)
    parser.add_argument('--ckpt_dir',           help='Checkpoint saving directory', type=str, required=True)
    parser.add_argument('--log_freq',           help='Logging frequency', type=int, required=True)
    parser.add_argument('--seed',               help='Universal random seed', type=int, default=42)
    parser.add_argument('--logname',            help='Name for log files', type=str, default='train_logs')
    
    args = parser.parse_args()
    
    print(args.eval)
    
    if args.eval:
        eval(args)
    else:
        train(args)
