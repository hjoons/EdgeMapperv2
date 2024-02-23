#!/bin/bash

mode=train
data_path=./sync
gt_path=./sync
filenames=nyudepthv2_train_files_with_gt_dense.txt
log_directory=./logs
checkpoint_path=./ckpt/checkpoint_100.pt
checkpoint_freq=50
log_freq=10
batch_size=60
num_epochs=200
learning_rate=0.0001
eval_path=./test
filenames_eval=nyudepthv2_test_files_with_gt.txt
num_threads=4

python runner.py --mode=$mode --data_path=$data_path --gt_path=$gt_path --filenames_file=$filenames --log_directory=$log_directory --checkpoint_path=$checkpoint_path --checkpoint_freq=$checkpoint_freq --log_freq=$log_freq --batch_size=$batch_size --num_epochs=$num_epochs --learning_rate=$learning_rate --gt_path_eval=$eval_path --filenames_file_eval=$filenames_eval --num_threads=$num_threads

