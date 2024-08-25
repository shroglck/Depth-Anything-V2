# minimum working example to load a single OXE dataset
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import argparse
from openai import AzureOpenAI
import re
import json
import os
ENDPOINT = "https://zhan-eastus-1.openai.azure.com"
IGNORE = ['right side', 'edge', 'left side', 'corner', 'side', 'cardboard fence']
pattern = r'\((.*?)\)'
UNDEFINED_TASKS = {
    'move the white can to the right side of the silver pot': [('white can', ''), ('silver pot', '')],
    'moved the orange cloth above the silver pan': [('orange cloth', ''), ('silver pan', '')],
    'remove the yellow brush in the silver pot': [('yellow brush', ''), ('silver pot', '')],
    'move spoon out of silver pot and placed on the table': [('spoon', ''), ('silver pot', ''), ('table', '')],
    'put lid on pot or pan': [('lid', ''), ('pot', ''), ('pan', '')],
    'take the green arch on the right and put it on top of the red arch': [('green arch', ''), ('red arch', '')]
}

task_desc = ("You will be given a sentence which are instructions for a robot to perform a certain task. "
             "Answer with just a list of the key objects that are part of the task. If there is positonal information DIRECTLY "
             "BEFORE the key object (i.e. bottom left burner, left pan, low drawer etc.), include the position information in "
             "parantheses. If an object has multiple related descriptions, don't separate them by commas. If the task is "
             "nonsensical, empty, or there are no objects, just answer with "
             "\"No key objects\". \n\nExamples:\n{in_context_examples}")
in_context_examples = '\n'.join(
    ["Task: unfold the cloth from top right to bottom left\nAnswer: cloth",
     "Task: take the blue spatula and put it on the right burner pushing the orange thing to the left burner\nAnswer: blue spatula, burner (right), orange thing, burner (left)",
     "Task: move the fork to the lower left side of the table\nAnswer: fork, table"]
)
task_desc = task_desc.format(in_context_examples=in_context_examples)

def call_chatgpt(chatgpt_messages, task, model="gpt-4o-mini-2024-07-18"):
    
    if task in UNDEFINED_TASKS:
        return UNDEFINED_TASKS[task]
    
    response = client.chat.completions.create(
        model=model, messages=chatgpt_messages
    )
    reply = response.choices[0].message.content
    
    if reply == 'No key objects':
        return []
    
    object_list = reply.split(',')
    object_list = list(set([obj.strip() for obj in object_list]))
    object_list = [obj for obj in object_list if not any([w in obj for w in IGNORE])]
    
    return_list = []
    
    for i in range(len(object_list)):
        if '(' in object_list[i] and ')' in object_list[i]:
            pos_word = re.findall(pattern, object_list[i])[0]
            object_list[i] = object_list[i].split('(')[0].strip()
            object = object_list[i]
            
            pos_idx = task.find(pos_word)
            correct = False
            
            while pos_idx != -1:
                
                object_start_idx = pos_idx + len(pos_word) + 1
                if task[object_start_idx: object_start_idx + len(object)] == object:
                    correct = True
                    break
                pos_idx = task.find(pos_word, pos_idx + 1)
            
            if correct:
                pos = pos_word
            else:
                pos = ''
            
        else:
            pos = ''
        
        return_list.append((object_list[i], pos))
    
    return list(set(return_list))

    
def prepare_chatgpt_message(task_prompt):
    system_message = task_desc
    messages = [{"role": "system", "content": system_message}]
    messages += [{"role": "user", "content": task_prompt}]
    return messages


def get_object_list(task):
    
    request = prepare_chatgpt_message(task)
    full_object_list = call_chatgpt(request, task)
    
    return full_object_list

def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--openai-api-key', type=str, default='')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    args = params()
    client = AzureOpenAI(
        api_key=args.openai_api_key,
        azure_endpoint=ENDPOINT,
        api_version="2023-05-15"
    )
    dataset = tfds.load('bridge_dataset', data_dir=args.data_dir, split='train')
    key_object_dict = {}

    for example in dataset:
        
        task = [d['language_instruction'].numpy().decode('utf-8') for d in example['steps'].take(1)][0].lower()

        if task in key_object_dict:
            print(f'Task: {task} already processed')
            continue
        
        return_list = get_object_list(task)
        
        if return_list == []:
            key_object_dict[task] = {
                'key_objects': [],
                'positions': []
            }
        else:
            key_object_list = [obj[0] for obj in return_list]
            pos_list = [obj[1] for obj in return_list]
            print(f'Task: {task}, Key objects: {key_object_list}, Positions: {pos_list}')
            print()
            
            key_object_dict[task] = {
                'key_objects': key_object_list,
                'positions': pos_list
            }
    
    with open(os.path.join(params.data_dir, 'bridge_dataset_seg', '1.0.0', 'key_objects.json'), 'w') as f:
        json.dump(key_object_dict, f, indent=6)