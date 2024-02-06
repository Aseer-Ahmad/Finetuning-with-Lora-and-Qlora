#helper.py
import torch
import torch
from torch.ao.quantization import (
  get_default_qconfig_mapping,
  get_default_qat_qconfig_mapping,
  QConfigMapping,
)
import torch.ao.quantization.quantize_fx as quantize_fx
import copy

from torch.quantization import per_channel_dynamic_qconfig
from torch.quantization import quantize_dynamic_jit

import random
import numpy as np
import psutil
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def check_gpu_memory():
	gpu_mem = torch.cuda.memory_allocated() / 1e6
	gpu_mem_max = torch.cuda.max_memory_allocated() / 1e6

	print(f"GPU Memory Allocated: {gpu_mem} MB")
	print(f"GPU Max Memory Allocated: {gpu_mem_max} MB")
	return gpu_mem, gpu_mem_max

def check_cpu_memory():
	cpu_memory = psutil.virtual_memory()
	cpu_mem    = cpu_memory.used / 1e6

	print(f"CPU Memory Used: {cpu_mem} MB")
	return cpu_mem


def save_checkpoint(model, optimizer, lr_scheduler, checkpoint_path ):
	
	torch.save({
        'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'lr_state_dict' : lr_scheduler.state_dict()
    }, checkpoint_path)

	#report checkpoint size on-disk
	size_in_bytes = os.path.getsize(checkpoint_path)
	print(f"{checkpoint_path} : {size_in_bytes} bytes")

	return size_in_bytes


def load_checkpoint(model, optimizer, lr_scheduler, checkpoint_path):
	checkpoint = torch.load(checkpoint_path)
	model.load_state_dict(checkpoint['model_state_dict'])

	if optimizer != None and lr_scheduler != None :
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		lr_scheduler.load_state_dict(checkpoint['lr_state_dict'])
	
	return model, optimizer, lr_scheduler

def FX_graph_mode_quantization(model, input):

	# post training static quantization
	example_batch = (input)
	model_to_quantize = copy.deepcopy(model)
	qconfig_mapping = get_default_qconfig_mapping("qnnpack")
	model_to_quantize.eval()
	# calibrate : pass through model_prepared example batch
	# model(**example_batch)
	model_prepared = quantize_fx.prepare_fx(model_to_quantize, qconfig_mapping, example_batch)
	model_quantized = quantize_fx.convert_fx(model_prepared)

	return model_quantized

def dynamic_quantization(model):
	quantized_model = torch.quantization.quantize_dynamic(
   						model, {torch.nn.Linear, torch.nn.Conv1d}, dtype=torch.qint8)
	
	return quantized_model


def static_quantization(model):
	model.eval()
	model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
	model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['conv', 'relu']])
	model_fp32_prepared = torch.ao.quantization.prepare(model_fp32_fused)
	# calibrate : pass through model_prepared example batch
	# model(**example_batch)
	model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
	return model_int8

def check_model_size(model):
	PARENT_PATH = os.getcwd()
	MODEL_PATH  = os.path.join(PARENT_PATH, 'temp.pth')
	torch.save({
		'model' : model.state_dict()
	}, MODEL_PATH)
	print('Model Size (MB):', os.path.getsize(MODEL_PATH)/1e6)
	os.remove(MODEL_PATH)


def metric1():
	pass
	
def metric2():
	pass
	
def getMemInfo():
	pass
	