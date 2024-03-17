import torch
import torch.nn as nn
import os
import sys
import argparse
from mobilenetv3 import MobileNetSkipConcat
import subprocess

def pytorch_to_onnx(pytorch_model: nn.Module, save_onnx: str):
	"""
	Converts pytorch model into ONNX runtime model
	
	Parameters:
		pytorch_model:torch.nn.Module - pytorch model to convert
		save_onnx:str - file name for saved ONNX model
		
	Returns:
		None
	"""
	model.eval()
	
	torch.onnx.export(model, torch.randn(1,3,480,640), save_onnx, export_params=True, opset_version=13, do_constant_folding=True)
	
def onnx_to_tensorrt(onnx_path: str, save_trt: str, enable_fp16: bool = False, enable_int8: bool = False):
	"""
	Converts ONNX model to TensorRT model
	
	Parameters:
		onnx_path:str - file name of onnx model
		save_trt:str - file name for saved TensorRT model
		enable_fp16:bool - enable fp16 quantization default=False
		enable_int8:bool - enable int8 quantization default=False
		
	Returns:
		bool - success
	"""
	command = ['trtexec', f'--onnx={onnx_path}', f'--saveEngine={save_trt}', '--sparsity=enable']
	if enable_fp16:
		command.append('--fp16')
	if enable_int8:
		command.append('--int8')
	
	result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	
	return result.returncode == 0
	
def pytorch_to_tensorrt(pytorch_model: nn.Module, save_trt: str, enable_fp16: bool = False, enable_int8: bool = False):
	"""
	Converts pytorch model into TensorRT model
	
	Parameters:
		pytorch_model:torch.nn.Module - pytorch model to convert
		save_trt:str - file name for saved TensorRT model
		enable_fp16:bool - enable fp16 quantization default=False
		enable_int8:bool - enable int8 quantization default=False
		
	Returns:
		bool - success
	"""
	pytorch_to_onnx(pytorch_model, 'temp.onnx')
	success = onnx_to_tensorrt('temp.onnx', save_trt)
	
	os.remove('temp.onnx')
	
	return success

