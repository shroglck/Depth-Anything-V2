# minimum working example to load a single OXE dataset
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import torch
from openai import AzureOpenAI
import argparse
import re
import pickle
import time
import json
import os
POS_PATTERN = r'\((.*?)\)'
REMOVE_POS_PATTERN = r'\s*\(.*?\)\s*'
ENDPOINT = "https://zhan-westus-0.openai.azure.com"
ORGANIZATION = "zhan-westus-0-global"
IGNORE = {'close', 'near', 'far', 'white', 'blueberry'}

task_desc = "You will be given a sentence which are instructions for a robot to perform a certain task. Answer with just a list of the key objects that are part of the task. If the key object has positional information (e.g top, middle, left, right), then include the positional information in parantheses. The positional information must be exact to be included, so descriptors like near, close, or far should not be included. If an object has multiple related descriptions, don't separate them by commas.\n\nExamples:\n{in_context_examples}"
in_context_examples = '\n'.join(["Task: place the grasped shiny green rxbar chocolate in the bottom drawer\nAnswer: shiny green rxbar chocolate, drawer (bottom)", "Task: move the middle Sprite can on top of the blue plate\nAnswer: Sprite can (middle), blue plate", "Task: Grab the green chip bag from the left table and put it in the white bowl. \nAnswer: green chip bag, table (left), white bowl"])
task_desc = task_desc.format(in_context_examples=in_context_examples)


def call_chatgpt(chatgpt_messages, model="gpt-4-1106-preview"):
    
    # Call ChatGPT to get the key objects
    response = client.chat.completions.create(
        model=model, messages=chatgpt_messages
    )
    reply = response.choices[0].message.content
    object_list = reply.split(',')
    object_list = [obj.strip() for obj in object_list]
    final_obj_list = []
    pos_list = []
    
    # Find all the positional words in the object list
    for object in object_list:
        pos_info = re.findall(POS_PATTERN, object)
        if len(pos_info) == 0 or pos_info[0] in IGNORE:
            pos_list.append('')
        else:
            pos_list.append(pos_info[0])
        
        final_obj_list.append(re.sub(REMOVE_POS_PATTERN, '', object))
        
    return final_obj_list, pos_list

    
def prepare_chatgpt_message(task_prompt):
    system_message = task_desc
    messages = [{"role": "system", "content": system_message}]
    messages += [{"role": "user", "content": task_prompt}]
    return messages


def get_object_list(task_batch):
    
    batched_requests = prepare_chatgpt_message(task_batch)
    full_object_list, position_list = call_chatgpt(batched_requests)
    
    return full_object_list, position_list

def add_index(example, index):
    
    example['idx'] = index['idx']
    
    return example

def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=0,
                        help='Shard of the dataset to save', choices=[i for i in range(1024)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--openai-key', type=str)
    parser.add_argument('--pickle_file_path', type=str, default='key_objects.pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':   
    
    params = params()
    num_gpus = torch.cuda.device_count()
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    dataset = tfds.load('fractal20220817_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.zip((dataset, data_idx))
    dataset = dataset.map(add_index, num_parallel_calls=1)
    
    client = AzureOpenAI(
        api_key=params.openai_key,
        azure_endpoint=ENDPOINT,
        api_version="2023-05-15"
    )
    
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    dataset = tfds.load('fractal20220817_depth_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))],
                 'timestep_length': [len(item['steps']) for item in dataset]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.zip((dataset, data_idx))
    dataset = dataset.map(add_index, num_parallel_calls=1)
    
    image_batch, object_list, instruction_batch = [], [], []
    object_dict = {}
    
    pos_vocab = set()
    
    print(f'Starting to extract key objects for shard {shard}...')
    start_time = time.time()
    
    for example in dataset:
        
        # Extract language task and example index
        task = [d['observation']['natural_language_instruction'].numpy().decode('utf-8') for d in example['steps'].take(1)][0]
        example_idx = int(example['idx'].numpy())
        
        # Get key objects and position words
        object_list, pos_list = get_object_list(task)
        print(object_list, pos_list)
        
        for pos in pos_list:
            if pos:
                pos_vocab.add(pos)
                
        object_dict[f'{example_idx}'] = {
            'objects': object_list,
            'positions': pos_list
        }
    
    # Save pickle file with key objects and positional words
    with open(params.pickle_file_path, 'wb') as f:
        pickle.dump(object_dict, f)
        
    # Update positional vocabulary
    with open(os.path.join(params.data_dir, 'fractal20220817_obj_data/0.1.0', f'pos_vocab_{shard}.json'), 'w') as f:
        json.dump(list(pos_vocab), f)
        
    print(f"Time taken: {time.time() - start_time}")