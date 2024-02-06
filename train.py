#train.py
from dataloader import getDataset, getDataloaders
from helpers.helper import check_cpu_memory, check_gpu_memory, save_checkpoint, load_checkpoint, set_seed, dynamic_quantization,check_model_size

import yaml
import os
import time
import sys
# import tensorflow as tf

from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from torch.optim import AdamW, Adam, SGD
from transformers import get_scheduler
# import evaluate

from tqdm.auto import tqdm

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler 

import pandas as pd

log_dir = "logs"  # Specify the directory where you want to store the logs
# summary_writer = tf.summary.create_file_writer(log_dir)

YAML_PATH = 'config.yaml'
PARENT_PATH  = os.getcwd()


def train(train_dataloader, trained_model_filename, yaml_data):

	print("\nin train")
	#arguments
	MODEL_NAME        = yaml_data['MODEL_NAME']
	NUM_EPOCHS        = int(yaml_data['NUM_EPOCHS'])
	LR         	      = float(yaml_data['LR'])
	SAVE_CHKPNT_EPOCH = yaml_data['SAVE_CHKPNT_EPOCH']
	MODEL_CHKPNT_DIR  = yaml_data['MODEL_CHKPNT_DIR']
	SEQ_LEN           = int(yaml_data['SEQ_LEN'])
	BATCH_SIZE		  = int(yaml_data['BATCH_SIZE'])
	OPTIMIZER_NAME    = yaml_data['OPTIMIZER_NAME']
	OPT_LVL 		  = yaml_data['OPT_LEVEL']
	PRECISION_TYPE    = yaml_data['PRECISION_TYPE']

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	print(f"MODEL_NAME : {MODEL_NAME}\nNUM_EPOCHS : {NUM_EPOCHS} \nLR : {LR}\nSAVE_CHKPNT_EPOCH : {SAVE_CHKPNT_EPOCH} \
	   \nMODEL_CHKPNT_DIR : {MODEL_CHKPNT_DIR}\nSEQ_LEN : {SEQ_LEN}\nBATCH_SIZE : {BATCH_SIZE}\nOPTIMIZER_NAME : {OPTIMIZER_NAME}\ndevice : {device}\nOPT_LEVEL : {OPT_LVL} \
	   \nPRECISION_TYPE:{PRECISION_TYPE}\n")
	
	num_batches = len(train_dataloader)
	step = 0
	epoch_total_time  = 0
	forward_total_time, backward_total_time = 0, 0
	running_loss = 0
	tot_gpu_mem = 0
	tot_cpu_mem = 0
	df_list     = []

	print(f"\nloading model {MODEL_NAME} , optimizer and scheduler")

	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME) # , low_cpu_mem_usage = True
	model.to(device)
	num_training_steps = NUM_EPOCHS * num_batches
	optimizer    = get_opt(model, OPTIMIZER_NAME, yaml_data)
	lr_scheduler = get_schdlr(optimizer, num_training_steps)
	scaler       = GradScaler(enabled = True)
	
	# #amp initialization
	# #avoiding as outputs.loss has unexpected behaviour 
	# model, optimizer = amp.initialize(model, optimizer, opt_level = OPT_LVL)

	if trained_model_filename != None:
		model_chkpnt = os.path.join(PARENT_PATH, yaml_data['MODEL_CHKPNT_DIR'], trained_model_filename)  
		model , optimizer, lr_scheduler = load_checkpoint(model, optimizer, lr_scheduler, model_chkpnt)	
		print(f'{MODEL_NAME} loaded from {model_chkpnt}')

	model.train()
	
	# progress_bar = tqdm(range(num_training_steps))
	print("\nmodel, opt, schdl loaded")
	print("\nbeginning training ...")

	for epoch in range(NUM_EPOCHS):
		
		start_time = time.time()

		try : 
			for i, batch in enumerate(train_dataloader):
				
				batch = {k: v.to(device) for k, v in batch.items()}

				if PRECISION_TYPE == 'MIXED':
					with autocast(dtype = torch.float16, enabled = True):
						forward_st = time.time()
						outputs = model(**batch)
						forward_et = time.time()
						forward_total_time += (forward_et - forward_st)
				elif PRECISION_TYPE == 'SINGLE':
						forward_st = time.time()
						outputs = model(**batch)
						forward_et = time.time()
						forward_total_time += (forward_et - forward_st)

				loss = outputs.loss

				running_loss += loss

				# #avoiding as outputs.loss has unexpected behaviour 
				# with amp.scale_loss(loss, optimizer) as scaled_loss:
				# 	scaled_loss.backward()

				backward_st = time.time()
				
				if PRECISION_TYPE == 'MIXED':
					scaler.scale(loss).backward() 
				elif PRECISION_TYPE == 'SINGLE':
					loss.backward()
				
				backward_et = time.time()
				backward_total_time += (backward_et - backward_st)

				if PRECISION_TYPE == 'MIXED':
					scaler.step(optimizer)
					scale = scaler.get_scale()
					scaler.update()
					skip_lr_shdlr = (scale > scaler.get_scale()) # to avoid lr step in case of NaN gradients

					if not skip_lr_shdlr : 
						lr_scheduler.step()
				
				elif PRECISION_TYPE == 'SINGLE':
					optimizer.step()
					lr_scheduler.step()

				optimizer.zero_grad()
				
				# progress_bar.update(1)

				step += 1

				print(f"epoch : {epoch+1} / {NUM_EPOCHS} iter : {i} / {num_batches},  loss : {loss}")
				# tf.summary.scalar('loss', loss, step = step)

				# gpu_mem, gpu_mem_max = check_gpu_memory()
				# cpu_mem              = check_cpu_memory()
				# tf.summary.scalar('gpu_mem', gpu_mem, step = step)
				# tf.summary.scalar('gpu_mem_max', gpu_mem_max, step = step)
				# tf.summary.scalar('cpu_mem', cpu_mem, step = step)
		
		except RuntimeError as e:
			print(f"RuntimeError : {e}")

		end_time = time.time()
		epoch_time = end_time - start_time
		epoch_total_time += epoch_time

		#average training loss
		avg_loss = running_loss / num_batches
		print(f"\nepoch : {epoch+1} / {NUM_EPOCHS} \naverage training loss : {avg_loss.item()}")
		running_loss = 0
		
		#training time per epoch
		print(f'total training time : {epoch_time:.2f} seconds')
		# tf.summary.scalar('epoch_exe_time', epoch_time, step = epoch)

		#throughput token : tokens processed per second
		t_tps = (SEQ_LEN * BATCH_SIZE * num_batches) / epoch_time
		print(f'tokens processed per second : {t_tps:.4f}')
		# tf.summary.scalar('token_throughput', t_tps, step = epoch)

		#throughput input: input sequence processed per second
		is_tps = (BATCH_SIZE * num_batches) / epoch_time
		print(f'input sequence of size {SEQ_LEN} processed per second : {is_tps:.4f}')

		gpu_mem, gpu_mem_max = check_gpu_memory()
		cpu_mem              = check_cpu_memory()

		tot_cpu_mem += cpu_mem
		tot_gpu_mem += gpu_mem

		print(f"lr schdl : {lr_scheduler.state_dict()}")

		#save checkpoint on disk 
		if SAVE_CHKPNT_EPOCH is not None and epoch % SAVE_CHKPNT_EPOCH == 0:
			checkpoint_path = os.path.join(PARENT_PATH, MODEL_CHKPNT_DIR, f'{MODEL_NAME}_{PRECISION_TYPE}_chkpoint_{epoch+1}.pth')
			size_in_bytes = save_checkpoint(model, optimizer, lr_scheduler, checkpoint_path)
	
		df_list.append([epoch+1, avg_loss.item() , epoch_time, t_tps, is_tps ])
		df = pd.DataFrame(df_list, columns = ['epoch' , 'loss', 'training time', 'token throughput', 'input throughput' ])
		df.to_csv( os.path.join(PARENT_PATH, log_dir, f'report_{MODEL_NAME}_{PRECISION_TYPE}.csv') , index = False)

	#total training time per epoch
	print(f'Total training Time for {NUM_EPOCHS} epoch : {epoch_total_time :.2f} seconds')

	#average training time per epoch
	print(f'Average training Time per epoch : {epoch_total_time / NUM_EPOCHS :.2f} seconds')

	#token throughput 
	print(f'token throughput : {(SEQ_LEN * BATCH_SIZE * num_batches * NUM_EPOCHS) / epoch_total_time :.4f} tokens per second')

	#input sequence throughput
	print(f'input sequence throughput : {(BATCH_SIZE * num_batches * NUM_EPOCHS) / epoch_total_time :.4f} input sequences per second')

	#average forward pass time per epoch
	print(f'Average forward pass Time per epoch : {forward_total_time / NUM_EPOCHS :.2f} seconds')

	#average backward pass time per epoch
	print(f'Average backward pass Time per epoch : {backward_total_time / NUM_EPOCHS :.2f} seconds')

	#average gpu memory consumption per epoch
	print(f'Average gpu memory consumption per epoch: {tot_gpu_mem / NUM_EPOCHS :.4f} MB')

	#average cpu memory consumption per epoch
	print(f'Average cpu memory consumption per epoch: {tot_cpu_mem / NUM_EPOCHS :.4f} MB')

	#maximum gpu memory consumed
	print(f'maximum gpu memory consumed : { gpu_mem_max:.4f} MB')


	return model

def get_opt(model, OPTIMIZER_NAME, yaml_data):

	#arguments
	LR           = float(yaml_data['LR'])
	WEIGHT_DECAY = float(yaml_data['WEIGHT_DECAY'])
	MOMENTUM     = float(yaml_data['MOMENTUM'])

	if OPTIMIZER_NAME == 'AdamW':
		optimizer = AdamW(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)
	elif OPTIMIZER_NAME == 'Adam':
		optimizer = Adam(model.parameters(), lr=LR, weight_decay = WEIGHT_DECAY)
	elif OPTIMIZER_NAME == 'SGD':
		optimizer = SGD(model.parameters(), lr=LR, momentum = MOMENTUM)

	return optimizer

def get_schdlr(optimizer, num_training_steps):
	
	lr_scheduler = get_scheduler(
    	name="linear", optimizer=optimizer,
		num_warmup_steps=0, 
		num_training_steps=num_training_steps
	)

	return lr_scheduler

def eval(model, eval_dataloader, model_chkpnt, quantize, yaml_data):
	
	print("starting eval ...")

	#arguments
	MODEL_NAME        = yaml_data['MODEL_NAME']
	MODEL_CHKPNT_DIR  = yaml_data['MODEL_CHKPNT_DIR']
	SEQ_LEN           = int(yaml_data['SEQ_LEN'])
	BATCH_SIZE		  = int(yaml_data['BATCH_SIZE'])

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	metric = None #evaluate.load("accuracy")	

	model_chkpnt = os.path.join(PARENT_PATH, MODEL_CHKPNT_DIR, model_chkpnt)

	if model is None and model_chkpnt != None:
		model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		model , optimizer, lr_scheduler,= load_checkpoint(model, None, None, model_chkpnt)

	if quantize : 
		print("model size before quantization")
		check_model_size(model)
		print(model)
		model = dynamic_quantization(model)
		device = torch.device("cpu")
		print(model)
		print("model size after quantization")
		check_model_size(model)

	model.to(device)
	model.eval()

	num_batches = len(eval_dataloader)
	count = 1
	running_loss = 0

	st = time.time()
	for batch in eval_dataloader:
		batch = {k: v.to(device) for k, v in batch.items()}
		with torch.no_grad():
			outputs 	= model(**batch)
			loss		= outputs.loss
			running_loss += loss

			print(f"batch : {count}/{num_batches} loss : {loss}")
			count += 1

			# logits      = outputs.logits
			# predictions = torch.argmax(logits, dim=-1)
			# metric.add_batch(predictions=predictions, references=batch["labels"])
			# metric.compute()

	et =  time.time()
	run_time = et - st

	gpu_mem, gpu_mem_max = check_gpu_memory()
	cpu_mem              = check_cpu_memory()

	#token throughput 
	print(f'total running time : {run_time} seconds')
	print(f'token throughput : {(SEQ_LEN * BATCH_SIZE * num_batches ) / run_time :.4f} tokens per second')
	print(f"average loss : {running_loss / num_batches}")

	check_model_size(model)
		

def config():
    # Read the YAML file
    with open(YAML_PATH, 'r') as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data

def free_memory():
	torch.cuda.empty_cache()

def loadModel(yaml_data):
	MODEL_NAME        = yaml_data['MODEL_NAME']
	print(f"\nloading model {MODEL_NAME}")
	model = AutoModelForCausalLM.from_pretrained(MODEL_NAME) # , low_cpu_mem_usage = True
	return model



def main():

	yaml_data  = config()
	print(yaml_data)
	
	SEED  = int(yaml_data['SEED'])

	set_seed(SEED)

	# set trained_model_filename if need to use a pretrained checkpoint ; else keep None
	# eg:  'gpt2_SINGLE_chkpoint_4.pth'
	trained_model_filename = 'gpt2_SINGLE_chkpoint_10.pth'  
	data = getDataset(yaml_data)
	train_dataloader, eval_dataloader = getDataloaders(data, yaml_data)
	model = train(train_dataloader, trained_model_filename,  yaml_data)
	# quantize = True
	# eval(None, eval_dataloader, trained_model_filename, quantize, yaml_data)

	# quantization attempt
	
	# model = loadModel(yaml_data)
	# MODEL_CHKPNT_DIR  = yaml_data['MODEL_CHKPNT_DIR']
	# checkpoint_path = os.path.join( PARENT_PATH, MODEL_CHKPNT_DIR, 'SINGLE', 'gpt2_SINGLE_chkpoint_7.pth')
	# model, _, _  = load_checkpoint(model, None, None, checkpoint_path)
	# model_quantize = dynamic_quantization(model)


if __name__ == '__main__':
	main()
