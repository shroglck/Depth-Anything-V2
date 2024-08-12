import tensorflow as tf
import tensorflow_datasets as tfds
# from openai import AzureOpenAI
import argparse
import re
import pickle
import time
import json
import os
import string
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize, sent_tokenize
lemmatizer = WordNetLemmatizer()
stop = set(stopwords.words('english') + list(string.punctuation))
IGNORE = {'place', 'close', 'knock', 'move', 'open'}
POSITIONAL_WORDS = {'left', 'right', 'top', 'bottom', 'front', 'back', 'middle'}
UNDEFINED_TASKS = {
    'pick green can': (['green can'], ['']),
    'move 7up can near sponge': (['7up can', 'sponge'], ['', '']),
    'move green can near sponge': (['green can', 'sponge'], ['', '']),
    '': ([], []),
}

undefined_tasks = set()

must_contain_alphe = True
min_term_len = 3

def noun_phrase_by_sentence(text):
    tokenized_content = []
    sentences = []
    for s in sent_tokenize(text):
        for ss in s.split('\n'):
            for sss in ss.split('     '):
                if len(sss.strip()) > 0:
                    sentences.append(sss.strip())


    np_list = []
    for sentence in sentences:
        np_sentence_list = []
        doc = nlp(sentence)
        for noun_chunk in doc.noun_chunks:
            sent_stop = [i for i in word_tokenize(noun_chunk.text.lower()) if i not in stop]
            # print ('sent_stop', sent_stop)
            cleaned_sent_stop = []
            for ss in sent_stop:
                if len(ss) >= min_term_len and any(c.isalpha() for c in ss):
                    cleaned_sent_stop.append(ss)

            # print ('cleaned_sent_stop', cleaned_sent_stop)
            lemmatized = []
            if len(cleaned_sent_stop) > 0:
                lemmatized=[lemmatizer.lemmatize(word) for word in cleaned_sent_stop]

            if len(lemmatized) > 0:
                # print ('lemmatized', lemmatized)
                for n in lemmatized:
                    np_sentence_list.append(n)
                np_sentence_list.append("_".join(lemmatized))

        if len(np_sentence_list) > 0:
            np_list.append(list(set(np_sentence_list)))
            # print('np_sentence_list', np_sentence_list)
    return np_list

def get_object_list(prompt):

    if prompt in UNDEFINED_TASKS:
        return UNDEFINED_TASKS[prompt]
    
    object_list = []
    position_list = []
    
    object_list = noun_phrase_by_sentence(prompt)[0]
    final_object_list = []
    
    for i in range(len(object_list)):
        
        word = object_list[i]
        # Check if the word is a subset of another word in the object list
        if not any([word in object_list[idx] and word != object_list[idx] for idx in range(len(object_list))]):
            
            if '_' in word:
                word = word.replace('_', ' ')
            for ignore_word in IGNORE:
                if ignore_word in word:
                    word = word.replace(ignore_word, '')
            
            if word:
                for pos_word in POSITIONAL_WORDS:
                    if pos_word in word:
                        position_list.append(pos_word)
                        word = word.replace(pos_word, '')
                        break
                else:
                    position_list.append('')
                
                final_object_list.append(word.strip())
    
    prompt_words = prompt.split(' ')
    for i in range(len(prompt_words)):
        if prompt_words[i] == 'can':
            previous_word = prompt_words[i - 1]
            if previous_word in final_object_list:
                final_object_list[final_object_list.index(previous_word)] = f'{previous_word} can'

    return final_object_list, position_list

def add_index(example, index):
    
    example['idx'] = index['idx']
    
    return example

def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=23,
                        help='Shard of the dataset to save', choices=[i for i in range(1024)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--pickle_file_path', type=str, default='key_objects.pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':   
    
    params = params()
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    dataset = tfds.load('fractal20220817_depth_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.zip((dataset, data_idx))
    dataset = dataset.map(add_index, num_parallel_calls=1)
    
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
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
        print(task, object_list, pos_list)
        
        # Add key objects to object dictionary
        object_dict[str(example_idx)] = {
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