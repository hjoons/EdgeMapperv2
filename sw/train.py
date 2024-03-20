import time
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import torch
import pandas as pd

from torch.optim.lr_scheduler import StepLR
from dataloader import get_loader
from h5dataloader import NewH5DataLoader
from mobilenetv3 import MobileNetSkipConcat
from GuidedDepth.model.loader import load_model
from utils.setup_funcs import init_logger, init_seeds
from eval import compute_errors
from loss import Depth_Loss

def train(args):
    init_seeds(args.seed)
    logging_prefix = args.logname
    logger = init_logger(f"{logging_prefix}/seed{args.seed}")
    device = (
        "cuda:1"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # split = 'eval' if args.eval else 'train'

    logger.info(f"Now using device: {device}")
    
    logger.info('Loading data...')
    
    train_loader = get_loader(args.train_path, args.batch_size, 'train')
    test_loader = get_loader(args.train_path, args.batch_size, 'eval')

    # train_loader = NewH5DataLoader(args, 'train') if ('.h5' in args.train_path or '.mat' in args.train_path) else NewDataLoader(args, 'train')
    # test_loader = NewH5DataLoader(args, 'eval') if ('.h5' in args.test_path or '.mat' in args.test_path) else NewDataLoader(args, 'eval')
    
    # model = MobileNetSkipConcat()
    model = load_model('GuideDepth', None)
    model = model.to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=15, gamma=0.1)
    
    start = 0
    
    if args.ckpt_path:
        logger.info('Loading checkpoint...')
        checkpoint = torch.load(args.ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['epoch']
    
    criterion = Depth_Loss(alpha=.1, beta=1, gamma=1, maxDepth=10.0)

    logger.info('Starting training...')
    model.train()
    
    train_loss = []
    test_loss = []
    for epoch in range(start, args.epochs):
        time_start = time.perf_counter()
        model = model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            image = batch['image']
            depth = batch['depth']
                        
            image = image.to(torch.device(device))
            depth = depth.to(torch.device(device))
            optimizer.zero_grad()
            
            pred = model(image)

            loss = criterion(pred, depth)

            cpu_loss = loss.cpu().detach().numpy()
            running_loss += cpu_loss
            
            if (batch_idx + 1) % args.log_freq == 0:
                logger.info(f'Batch {batch_idx + 1} / {len(train_loader)} || Loss: {running_loss / (batch_idx + 1)}')

            loss.backward()

            optimizer.step()

        train_loss.append(running_loss / len(train_loader))
        if (epoch + 1) % args.ckpt_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            ckpt_path = os.path.join(args.ckpt_dir, 'checkpoint_{}.pt'.format(epoch+1))
            torch.save(checkpoint, ckpt_path)
        
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
                
                loss = criterion(pred, depth)

                errors.append(compute_errors(depth, pred))

                cpu_loss = loss.cpu().detach().numpy()
                running_test_loss += cpu_loss
                
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

            test_loss.append(running_test_loss / len(test_loader))
            time_end = time.perf_counter()
            current_lr = optimizer.param_groups[0]['lr']

            logger.info(f'epoch: {epoch + 1} train loss: {running_loss / len(train_loader)} testing loss: {running_test_loss / len(test_loader)} learning rate: {current_lr} time: {time_end - time_start}')
            
            logger.info(f'abs_rel: {abs_rel}\nsq_rel: {sq_rel}\nrmse: {rmse}\nrmse_log: {rmse_log}\nd1: {d1}\nd2: {d2}\n d3: {d3}\n')
            if (epoch + 1) % args.ckpt_freq == 0:
                train_csv_path = os.path.join(args.ckpt_dir, 'train_loss_{}.csv'.format(epoch + 1))
                test_csv_path = os.path.join(args.ckpt_dir, 'test_loss_{}.csv'.format(epoch + 1))  

                train_df = pd.DataFrame(train_loss)
                test_df = pd.DataFrame(test_loss)

                train_df.to_csv(train_csv_path, header=False)
                test_df.to_csv(test_csv_path, header=False)
        
        scheduler.step()


if __name__ == '__main__':
    pass
