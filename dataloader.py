#dataloader.py
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2LMHeadModel
from datasets import disable_caching

from torch.utils.data import DataLoader


# comment disable_caching , when sure that 
# data pre-processing is working well
# disable_caching()

def getDataloaders(data, yaml_data):

	#arguments
	BATCH_SIZE = int(yaml_data['BATCH_SIZE'])

	data.set_format("torch")

	train_dataloader = DataLoader(data['train'], shuffle=True, batch_size=BATCH_SIZE)
	eval_dataloader  = DataLoader(data['test'], batch_size=BATCH_SIZE)

	print(f"trainDataLoader size : {len(train_dataloader)} batches of size {BATCH_SIZE}")
	print(f"evalDataLoader size  : {len(eval_dataloader)}  batches of size {BATCH_SIZE}")

	return train_dataloader, eval_dataloader

def getDataset(yaml_data):

	print(f'\nin getDataset')
	#arguments
	DATASET_NAME = yaml_data['DATASET_NAME']
	SEED         = int(yaml_data['SEED'])
	TEST_PER     = float(yaml_data['TEST_PER'])
	TOKENIZER    = yaml_data['TOKENIZER']
	SEQ_LEN      = int(yaml_data['SEQ_LEN'])

	#data and tokenizer
	data = load_dataset(DATASET_NAME)
	tokenizer = getTokenizer(TOKENIZER)	

	#split data
	data = data["train"].train_test_split(test_size=TEST_PER, seed=SEED)
	
	# print_data(data, 'train', 9)

	data = data.map( preprocess,
				 	# batched = True,
					# num_proc = 4,
				 	fn_kwargs = {'tokenizer' : tokenizer},
					remove_columns = data['train'].column_names
					)

	# print(data['train'][9])


	# data = data.map(lambda row : tokenizer(row['instruction']),
				 	# batched = True,
					# num_proc = 4,
					# remove_columns = data['train'].column_names
					# )

	# print(data)
	# print(data['train'][9])
	# print(tokenizer.decode(data["train"][9]["input_ids"]))

	lm_dataset = data.map(group_texts, 
					   	batched=True,
						num_proc=4,
						fn_kwargs = {'block_size' : SEQ_LEN} )
	print(lm_dataset)
	# print(lm_dataset['train'][0])

	# del data
	# del tokenizer

	return lm_dataset

def getTokenizer(TOKENIZER):
	tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
	tokenizer.pad_token = tokenizer.eos_token
	return tokenizer
	

def get_data_details(data):
	pass
	

def preprocess(data_row, tokenizer):
	return tokenizer(data_row['context'] + " " +   data_row['instruction'] + " " + data_row['response'],
					padding = True,
					truncation  = True)

def group_texts(examples, block_size):
	# Concatenate all texts.
	concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
	total_length = len(concatenated_examples[list(examples.keys())[0]])
	# We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
	# customize this part to your needs.

	# if total_length >= block_size:
	total_length = (total_length // block_size) * block_size
	
	# Split by chunks of block_size.
	result = {
		k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
		for k, t in concatenated_examples.items()
	}

	# labels because the model expects the argument to be named labels
	result["labels"] = result["input_ids"].copy()
	# del result['input_ids']
	return result

def print_data(data, split, index):
	print(f"instruction : {data[split]['instruction'][index]}")
	print(f"context     : {data[split]['context'][index]}")
	print(f"response    : {data[split]['response'][index]}")
	print(f"category    : {data[split]['category'][index]}")
	


		

