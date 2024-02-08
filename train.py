# !pip install "peft==0.2.0"
# pip install loralib
# !pip install "transformers==4.27.2" "datasets==2.9.0" "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1"  --upgrade --quiet

#train.py
from dataloader import getDataset, getDataloaders
from helpers.helper import check_cpu_memory, check_gpu_memory, save_checkpoint, load_checkpoint, set_seed, dynamic_quantization,check_model_size

import yaml
import os
import time
import sys
# import tensorflow as tf

from transformers import AutoModelForCausalLM,  TrainingArguments, Trainer , BitsAndBytesConfig
from torch.optim import AdamW, Adam, SGD
from transformers import get_scheduler
# import evaluate

from tqdm.auto import tqdm

import torch
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler 

import pandas as pd
from helpers.peft_prep import getLoraModel

log_dir = "logs"  # Specify the directory where you want to store the logs
# summary_writer = tf.summary.create_file_writer(log_dir)

YAML_PATH = 'config.yaml'
PARENT_PATH  = os.getcwd()


def train(data,  trained_model_filename, yaml_data):

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
	WEIGHT_DECAY      = yaml_data['WEIGHT_DECAY']
	PEFT_TYPE         = yaml_data['PEFT_TYPE']

	device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
	
	print(f"MODEL_NAME : {MODEL_NAME}\nNUM_EPOCHS : {NUM_EPOCHS} \nLR : {LR}\nSAVE_CHKPNT_EPOCH : {SAVE_CHKPNT_EPOCH} \
	   \nMODEL_CHKPNT_DIR : {MODEL_CHKPNT_DIR}\nSEQ_LEN : {SEQ_LEN}\nBATCH_SIZE : {BATCH_SIZE}\nOPTIMIZER_NAME : {OPTIMIZER_NAME}\ndevice : {device}\nPEFT_TYPE : {PEFT_TYPE} \
	   \nPRECISION_TYPE:{PRECISION_TYPE}\n")

	# num_batches = len(train_dataloader)
	step = 0
	epoch_total_time  = 0
	forward_total_time, backward_total_time = 0, 0
	running_loss = 0
	tot_gpu_mem = 0
	tot_cpu_mem = 0
	df_list     = []

	print(f"\nloading model {MODEL_NAME} , optimizer and scheduler")

	# num_training_steps = NUM_EPOCHS * num_batches
	# optimizer    = get_opt(model, OPTIMIZER_NAME, yaml_data)
	# lr_scheduler = get_schdlr(optimizer, num_training_steps)
	
	#LoRA / qLoRA 
	if PEFT_TYPE == 'lora':
		model = AutoModelForCausalLM.from_pretrained(MODEL_NAME) # low_cpu_mem_usage = True
		model = getLoraModel(model)
		model.to(device)
	elif PEFT_TYPE == 'qlora' :
		nf4_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_use_double_quant=True,
			bnb_4bit_compute_dtype=torch.bfloat16
		)
		model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=nf4_config, device_map="auto")
		model = getLoraModel(model)
	else:  # regular 
		model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
		model.to(device)

	if trained_model_filename != None:
		model_chkpnt = os.path.join(PARENT_PATH, yaml_data['MODEL_CHKPNT_DIR'], trained_model_filename)  
		model , optimizer, lr_scheduler = load_checkpoint(model, optimizer, lr_scheduler, model_chkpnt)	
		print(f'{MODEL_NAME} loaded from {model_chkpnt}')

	model.train()
	
	training_args = TrainingArguments(
		output_dir="gpt2_lora",
		evaluation_strategy="epoch",
		learning_rate=LR,
		weight_decay=WEIGHT_DECAY, 
		num_train_epochs = 1,
		save_strategy = "no",
		remove_unused_columns=False,
		per_device_train_batch_size  = BATCH_SIZE,
		per_device_eval_batch_size  = BATCH_SIZE
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=data["train"],
		eval_dataset=data["test"]
	)
	
	train_dataset_length = len(trainer.train_dataset)
	num_batches = (train_dataset_length + BATCH_SIZE - 1) // BATCH_SIZE

	print(f"num batches : {num_batches}\ntotal train samples : {train_dataset_length}")

	print("\nmodel, opt, schdl loaded")
	print("\nbeginning training ...")
		
	for epoch in range(NUM_EPOCHS):
		
		start_time = time.time()

		trainer.train()
		# print(trainer.get_optimizer_cls_and_kwargs(training_args)[1]['lr'])
		# last_updated_lr = trainer.get_optimizer_cls_and_kwargs(training_args)[1]['lr']
		# training_args.learning_rate = last_updated_lr
		# train_loss = trainer.evaluate()["loss"]
		# print(trainer.state.log_history)

		end_time = time.time()
		epoch_time = end_time - start_time
		epoch_total_time += epoch_time

		#average training loss
		# print(f"\nepoch : {epoch+1} / {NUM_EPOCHS} \naverage training loss : {train_loss}")
		
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
	
		# save csv file with model logs
		# df_list.append([epoch+1, avg_loss.item() , epoch_time, t_tps, is_tps ])
		# df = pd.DataFrame(df_list, columns = ['epoch' , 'loss', 'training time', 'token throughput', 'input throughput' ])
		# L_DIR = os.path.join(PARENT_PATH, log_dir)
		# if not os.path.exits(L_DIR):
		# 	os.makedirs(L_DIR)
		# df.to_csv( os.path.join(L_DIR, f'report_{MODEL_NAME}_{PRECISION_TYPE}.csv') , index = False)

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
	trained_model_filename = None
	data = getDataset(yaml_data)
	model = train(data,  trained_model_filename,  yaml_data)


if __name__ == '__main__':
	main()
