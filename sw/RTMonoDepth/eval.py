import torch
import argparse
import random
from copy import deepcopy
from mobilenetv3 import MobileNetSkipConcat
from dataloader import NewDataLoader

def compute_errors(gt, pred, epsilon=1e-6):
    """
    Compute error metrics between ground truth and predicted depth maps.

    Args:
        gt (torch.Tensor): Ground truth depth map.
        pred (torch.Tensor): Predicted depth map.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        Tuple: Tuple containing various error metrics.
    """
    # Ensure non-zero and non-negative ground truth values
    gt = gt.float().to('cpu')
    pred = pred.float().to('cpu')

    gt = torch.clamp(gt, min=epsilon)
    pred = torch.clamp(pred, min=epsilon)  # Also ensure predictions are positive

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < (1.25 ** 2)).float().mean()
    a3 = (thresh < (1.25 ** 3)).float().mean()

    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def evaluate(model, dataloader):
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    model = model.to(device)
    model.eval()
    
    errors = []

    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader.data):
            # print(f'{i + 1} / 100')
            image = sample['image']
            truth = sample['depth']
            # if (truth.min() < 200 and truth.min() >= 100):
                # break
            truth = torch.clamp(truth, 10, 1000)

            image = image.to(torch.device(device))
            truth = truth.to(torch.device(device))

            pred = model(image)
            pred = pred.squeeze(0)

            errors.append(compute_errors(truth, pred))
            if batch_idx == 1000:
                break

        error_tensors = [torch.tensor(e).to(device) for e in errors]

        error_stack = torch.stack(error_tensors, dim=0)

        mean_errors = error_stack.mean(0).cpu().numpy()

        abs_rel = mean_errors[0]
        sq_rel = mean_errors[1]
        rmse = mean_errors[2]
        rmse_log = mean_errors[3]
        a1 = mean_errors[4]
        a2 = mean_errors[5]
        a3 = mean_errors[6]
        print(f'abs_rel: {abs_rel}\nsq_rel: {sq_rel}\nrmse: {rmse}\nrmse_log: {rmse_log}\na1: {a1}\na2: {a2}\na3: {a3}\n')

def get_model(model_path: str, device):
    model = MobileNetSkipConcat().to(torch.device(device))
    ckpt = torch.load(model_path)
    model.load_state_dict(ckpt['model_state_dict'])
    return model

def get_train_loader(args):
    args_copy = deepcopy(args)
    args_copy.mode = 'train'
    train_loader = NewDataLoader(args_copy)
    return train_loader

def get_eval_loader(args):
    args_copy = deepcopy(args)
    args_copy.mode = 'eval'
    test_loader = NewDataLoader(args_copy, args_copy.mode)
    return test_loader

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',                      type=str,   help='train or test', default='train')
    parser.add_argument('--data_path',                 type=str,   help='path to the data', required=True)
    parser.add_argument('--gt_path',                   type=str,   help='path to the groundtruth data', required=True)
    parser.add_argument('--filenames_file',            type=str,   help='path to the filenames text file', required=True)
    parser.add_argument('--use_ckpt',                              help='if set, will use specified checkpoint', action='store_true')

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
    

    args = parser.parse_args()

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    eval_loader = get_eval_loader(args)

    model = get_model(args.checkpoint_path, device)
    evaluate(model, eval_loader)
