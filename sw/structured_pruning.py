import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from copy import deepcopy
import torch_pruning as tp

from losses import ssim, depth_loss
from dataloader import NewDataLoader
from RTMonoDepth.mobilenetv3 import MobileNetSkipConcat
from RTMonoDepth.model import MonoDepth
from utils.setup_funcs import init_logger, init_seeds

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
    """
    Evaluate a federated model on a dataset.

    Args:
        eval_dir (str): Path to the dataset for evaluation.

    Returns:
        None
    """
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

    random_indices = set()

    model.eval()
    with torch.no_grad():
        for batch_idx, sample in enumerate(dataloader.data):
            # print(f'{i + 1} / 100')
            image = sample['image']
            truth = sample['depth']
            # if (truth.min() < 200 and truth.min() >= 100):
                # break
            truth = torch.clamp(truth, 10, 1000)

            truth = truth.transpose(3,2).transpose(2,1)

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
    model = MobileNetSkipConcat()
    model.load_state_dict(torch.load(f'{model_path}', map_location=torch.device(device)))
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

def prune_model(model, eval_loader, train_loader, logger, pruning_fraction = 0.0):
    print(f"Pruning...")
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    ) 
    imp = tp.importance.MagnitudeImportance(p=1)
    pruning_iterations = 5
    model = model.to(device)
    example_inputs = torch.randn(1, 3, 192, 256).to(torch.device(device))
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs)

    ignored_layers = []
    for m in model.modules():
        if isinstance(m, torch.nn.Linear):
            ignored_layers.append(m)
    
    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs,
        importance = imp,
        iterative_steps = pruning_iterations,
        pruning_ratio =  pruning_fraction,
        ignored_layers = ignored_layers
    )
    for i in range(pruning_iterations):
        pruner.step()
        logger.info(f'Before fine-tuning: Iteration {i + 1} / {pruning_iterations}')
        test_model(model, eval_loader, logger, pruning_fraction)
        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        logger.info("Fine-tuning...")
        train(model, train_loader, logger)
        logger.info(f'After fine-tuning: Iteration {i + 1} / {pruning_iterations}')
        test_model(model, eval_loader, logger, pruning_fraction)
        # print(model)
    logger.info("Done pruning")
    logger.info("Saving pruned model...")
    torch.save(model.state_dict(), f'pruned_model_{pruning_fraction}.pt')
    return model

def train(model, train_loader, logger):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Now using device: {device}")
    # writer = SummaryWriter(args.log_directory)

    optimizer = torch.optim.AdamW(model.parameters())
    
    start = 0

    l1_criterion = torch.nn.L1Loss()

    logger.info('Starting training...')
    model.train()
    
    train_loss = []
    test_loss = []
    for epoch in range(start, 5):
        time_start = time.perf_counter()
        model = model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(train_loader.data):
            image = batch['image']
            depth = batch['depth']

            image = image.to(torch.device(device))
            depth = depth.to(torch.device(device))
            optimizer.zero_grad()
            
            pred = model(image)
            
            l1_loss = l1_criterion(pred, depth)
            
            ssim_loss = torch.clamp(
                (1 - ssim(pred, depth, 1000.0 / 10.0)) * 0.5,
                min=0,
                max=1,
            )
            
            gradient_loss = depth_loss(depth, pred, device=device)
            
            net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * torch.mean(gradient_loss))
                + (0.1 * torch.mean(l1_loss))
            )
            
            cpu_loss = net_loss.cpu().detach().numpy()
            running_loss += cpu_loss

            if ((batch_idx + 1) % 100 == 0):
                logger.info(f'Batch {batch_idx + 1} / {len(train_loader.data)} || Loss: {running_loss / (batch_idx + 1)}')

            net_loss.backward()

            optimizer.step()
        logger.info(f'epoch: {epoch + 1} train loss: {running_loss / len(train_loader)}')
        train_loss.append(running_loss / len(train_loader))

    # writer.close()
        
def test_model(model, test_loader, logger, prune_ratio):
    test_loss = []
    model.eval()
    l1_criterion = torch.nn.L1Loss()
    with torch.no_grad():
        running_test_loss = 0
        time_start = time.perf_counter()
        for batch_idx, batch in enumerate(test_loader.data):
            image = batch['image']
            depth = batch['depth']

            image = image.to(torch.device(device))
            depth = depth.to(torch.device(device))
            
            pred = model(image)

            # calculating the losses
            l1_loss = l1_criterion(pred, depth)

            ssim_loss = torch.clamp(
                (1 - ssim(pred, depth, 1000.0 / 10.0)) * 0.5,
                min=0,
                max=1,
            )

            gradient_loss = depth_loss(depth, pred, device=device)

            net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * torch.mean(gradient_loss))
                + (0.1 * torch.mean(l1_loss))
            )
            
            cpu_loss = net_loss.cpu().detach().numpy()
            running_test_loss += cpu_loss
        test_loss.append(running_test_loss / len(test_loader))
        time_end = time.perf_counter()

        logger.info(f'pruning ratio: {prune_ratio} testing loss: {running_test_loss / len(test_loader)} time: {time_end - time_start}')

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
    parser.add_argument('--logname',                   type=str,   help='name of the log', default='train_logs')

    args = parser.parse_args()
    init_seeds(args.seed)
    logging_prefix = args.logname
    logger = init_logger(f"{logging_prefix}/seed{args.seed}")

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    eval_loader = get_eval_loader(args)
    train_loader = get_train_loader(args)

    prune_vals = [0, 0.05, 0.1, 0.2, 0.3, 0.4]

    for value in prune_vals:
        model = get_model(f'{args.checkpoint_path}', device)
        prune_model(model, eval_loader, train_loader, logger, pruning_fraction=value)
        # break