import torch
import numpy as np

from loss import Depth_Loss
from mobilenetv3 import MobileNetSkipConcat
from dataloader import get_loader
from h5dataloader import NewH5DataLoader
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
        
def eval(args):
    init_seeds(args.seed)
    logging_prefix = args.logname
    logger = init_logger(f"{logging_prefix}/seed{args.seed}")
    device = (
        "cuda:0"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    logger.info(f"Now using device: {device}")
    
    logger.info('Loading data...')
    
    test_loader = get_loader(args.train_path, args.batch_size, 'eval')

    # test_loader = NewH5DataLoader(args, 'eval') if ('.h5' in args.test_path or '.mat' in args.test_path) else NewDataLoader(args, 'eval')
    model = MobileNetSkipConcat()
    model = model.to(torch.device(device))
            
    if args.ckpt_path:
        logger.info('Loading checkpoint...')
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)
    
    logger.info('Starting Evaluation...')
    
    model.eval()
    with torch.no_grad():
        running_test_loss = 0
        errors = []
        for batch_idx, batch in enumerate(test_loader):
            image = batch['image']
            depth = batch['depth']

            image = image.to(torch.device(device))
            depth = depth.to(torch.device(device))
            
            pred = model(image)

            pred = 10.0 / pred
            pred = torch.clamp(pred, 10.0 / 100.0, 10.0)
            
            loss = criterion(pred, depth)
            
            errors.append(compute_errors(depth, pred))

            cpu_loss = loss.cpu().detach().numpy()
            running_test_loss += cpu_loss
            
            if (batch_idx + 1) % (len(test_loader) // 4) == 0:
                logger.info(f'{batch_idx + 1} / {len(test_loader)}')
            
        error_tensors = [torch.tensor(e).to(device) for e in errors]
        error_stack = torch.stack(error_tensors, dim=0)
        mean_errors = error_stack.mean(0).cpu().numpy()
        
        abs_rel = mean_errors[0]
        sq_rel = mean_errors[1]
        rmse = mean_errors[2]
        rmse_log = mean_errors[3]
        d1 = mean_errors[4]
        d2 = mean_errors[5]
        d3 = mean_errors[6]
        
        logger.info(f'abs_rel: {abs_rel}\nsq_rel: {sq_rel}\nrmse: {rmse}\nrmse_log: {rmse_log}\nd1: {d1}\nd2: {d2}\n d3: {d3}\n')
