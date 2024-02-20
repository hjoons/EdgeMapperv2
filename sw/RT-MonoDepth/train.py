import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import numpy as np

from losses import ssim, depth_loss
from dataloader import NewDataLoader
from mobilenetv3 import MobileNetSkipConcat
from model import MonoDepth
from utils.setup_funcs import init_logger, init_seeds

def train(args):
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
    logger.info(f"Now using device: {device}")
    # writer = SummaryWriter(args.log_directory)

    logger.info('Loading data...')
    train_loader = NewDataLoader(args, args.mode)
    test_loader = NewDataLoader(args, args.mode)

    model = MonoDepth().to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    start = 0
    
    if args.use_ckpt:
        logger.info('Loading checkpoint...')
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start = checkpoint['epoch']

    l1_criterion = torch.nn.L1Loss()

    logger.info('Starting training...')
    model.train()
    
    train_loss = []
    test_loss = []
    for epoch in range(start, args.num_epochs):
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

        train_loss.append(running_loss / len(train_loader))
        if ((epoch + 1) % args.checkpoint_freq == 0):
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            ckpt_path = os.path.join(args.checkpoint_path, 'checkpoint_{}.pt'.format(epoch+1))
            torch.save(checkpoint, ckpt_path)

        model.eval()
        with torch.no_grad():
            running_test_loss = 0
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

            # if ((epoch + 1) % args.log_freq == 0):
            #     writer.add_scalar('Loss/train', (running_loss / len(train_loader)), global_step=(epoch + 1))
            #     writer.add_scalar('Loss/test', (running_loss / len(test_loader)), global_step=(epoch + 1))

            logger.info(f'epoch: {epoch + 1} train loss: {running_loss / len(train_loader)} testing loss: {running_test_loss / len(test_loader)} time: {time_end - time_start}')
    # writer.close()

if __name__ == '__main__':
    pass
