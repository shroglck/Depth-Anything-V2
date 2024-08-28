import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import numpy as np
import json
import argparse
import pickle
from collections import defaultdict
import time

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_list_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _int64_list_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def get_record_features():
    feature_dict = {}
    
    for raw_record in record_dataset.take(1):
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        
        for key, feature in example.features.feature.items():
            
            # Determine the data type based on which list is populated
            if feature.HasField('bytes_list'):
                data_type = 'byte' if len(feature.bytes_list.value) == 1 else 'bytes list'
            elif feature.HasField('float_list'):
                data_type = 'float' if len(feature.float_list.value) == 1 else 'float list'
            elif feature.HasField('int64_list'):
                data_type = 'int64' if len(feature.int64_list.value) == 1 else 'int64 list'
            else:
                data_type = 'unknown'
            
            feature_dict[key] = data_type

    feature_dict['key_objects'] = 'bytes list'
    feature_dict['positional_words'] = 'bytes list'
    return feature_dict


def serialize_example(example):
    
    feature = {}
    
    for key in ['aspects', 'attributes']:
        
        for key_2 in list(example[key].keys()):
            
            feature_name = '/'.join([key, key_2])
            feature[feature_name] = method_dict[feature_dict[feature_name]](example[key][key_2])
    
    steps_dict = defaultdict(list)
    for item in example['steps']:
        
        for steps_key in item:
            
            if steps_key not in ['action', 'observation']:
                steps_dict['/'.join(['steps', steps_key])].append(item[steps_key])
            else:
                
                for key_2 in item[steps_key]:
                    
                    if key_2 in ['depth', 'image']:
                        steps_dict['/'.join(['steps', steps_key, key_2])].append(tf.io.encode_png(item[steps_key][key_2]).numpy())
                    elif key_2 == 'natural_language_instruction':
                        steps_dict['/'.join(['steps', steps_key, key_2])].append(item[steps_key][key_2].numpy())
                    else:
                        tensor = item[steps_key][key_2]
                        flattened_tensor = tf.reshape(tensor, [-1]).numpy().tolist()
                        steps_dict['/'.join(['steps', steps_key, key_2])].extend(flattened_tensor)
    
    for key in steps_dict:
        feature[key] = method_dict[feature_dict[key]](steps_dict[key])
    
    key_objects_bytes = [ko.numpy() for ko in example['key_objects']]
    positional_words_bytes = [pw.numpy() for pw in example['positional_words']]
    feature['key_objects'] = method_dict[feature_dict['key_objects']](key_objects_bytes)
    feature['positional_words'] = method_dict[feature_dict['positional_words']](positional_words_bytes)
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def write_tfrecord(dataset):
    
    tfrecord_file = os.path.join(
        params.obj_data_dir,
        f'fractal20220817_obj_data-train.tfrecord-{shard_str}-of-01024'
    )
    
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for example in dataset:
            serialized_example = serialize_example(example)
            writer.write(serialized_example)


def add_index(dataset):
    
    def map_index(example, index):
        example['idx'] = index['idx']
        return example
    
    # Apply the function to each (example, index) pair in the dataset
    data_dict = {'idx': [idx for idx in range(len(dataset))]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset_with_idx = tf.data.Dataset.zip((dataset, data_idx))
    dataset_with_idx = dataset_with_idx.map(map_index, num_parallel_calls=1)

    return dataset_with_idx

def add_object_list(example):
        
    def py_func(idx):
        
        idx_str = tf.strings.as_string(idx)
        return tf.convert_to_tensor(data[idx_str.numpy().decode('utf-8')]['objects'], dtype=tf.string), \
                tf.convert_to_tensor(data[idx_str.numpy().decode('utf-8')]['positions'], dtype=tf.string)
    
    # Use tf.py_function to call py_func, ensuring to specify the output types correctly
    key_object_list_str, key_position_list_str = tf.py_function(
        py_func,
        [example['idx']],
        [tf.string, tf.string]
    )
    
    # No need to cast since output is already tf.string
    example['key_objects'] = key_object_list_str
    example['positional_words'] = key_position_list_str
    
    return example

def save_dataset_info():
    
    features_path = os.path.join(
        params.data_dir,
        'fractal20220817_data',
        '0.1.0',
        'features.json'
    )
    dset_info_path = os.path.join(
        params.data_dir,
        'fractal20220817_data',
        '0.1.0',
        'dataset_info.json'
    )
    
    with open(features_path, 'r') as f:
        features = json.load(f)
    with open(dset_info_path, 'r') as f:
        dset_info = json.load(f)
    
    dset_info['name'] = 'fractal20220817_obj_data'
    key_objects_dict = {
        'tensor': {
            'dtype': 'bytes',
            'shape': {'dimensions': [-1]}
        },
        'pythonClassName': 'tensorflow_datasets.core.features.tensor_feature.Tensor'
    }
    features['featuresDict']['features']['key_objects'] = key_objects_dict
    features['featuresDict']['features']['positional_words'] = key_objects_dict
    
    with open(os.path.join(params.obj_data_dir, 'features.json'), 'w') as f:
        json.dump(features, f, indent=6)
    with open(os.path.join(params.obj_data_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dset_info, f, indent=6)

def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=1,
                        help='Shard of the dataset to save', choices=[i for i in range(1025)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--obj-data-dir', type=str, default='/data/shresth/octo-data/fractal20220817_obj_data/0.1.0')
    parser.add_argument('--pickle_file_path', type=str, default='key_objects.pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    params = params()
    
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    # Load pickle file and dataset/record dataset
    dataset = tfds.load('fractal20220817_data:0.1.0', data_dir=params.data_dir,
                        split=split)
    with open(params.pickle_file_path, 'rb') as f:  
        data = pickle.load(f)

    # Remove pickle file
    os.remove(params.pickle_file_path)
    
    record_dataset = tf.data.TFRecordDataset(
        os.path.join(
            params.data_dir, 'fractal20220817_data', '0.1.0',
            f'fractal20220817_data-train.tfrecord-{shard_str}-of-01024'
        )
    )
    
    start_time = time.time()
    print(f'Saving key objects for shard {shard}...')
    dataset = add_index(dataset)
    dataset = dataset.map(add_object_list)
    
    method_dict = {
        'byte': _bytes_feature,
        'bytes list': _bytes_list_feature,
        'float': _float_feature,
        'float list': _float_list_feature,
        'int64': _int64_feature,
        'int64 list': _int64_list_feature
    }
    feature_dict = get_record_features()
    print('Serializing and writing dataset to tfrecord...')
    write_tfrecord(dataset)
    print('Updating feature and info dictionary...')
    save_dataset_info()
    print(f'Finished saving key objects for shard {shard} in {time.time() - start_time:.2f} seconds')