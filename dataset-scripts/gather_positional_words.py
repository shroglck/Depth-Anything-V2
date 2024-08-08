import os
import json
import argparse

def params():
        
    parser = argparse.ArgumentParser(description='Save the positional words')
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    config = params()
    data_dir = config.data_dir
    full_pos_list = set()
    
    for i in range(1025):
        shard = i
        
        with open(os.path.join(data_dir, 'fractal20220817_obj_data', '0.1.0', f'pos_vocab_{shard}.json'), 'r') as f:
            features = json.load(f)
            features = set(features)
        
        # Remove temporary files
        os.remove(os.path.join(data_dir, 'fractal20220817_obj_data', '0.1.0', f'pos_vocab_{shard}.json'))
        
        full_pos_list = full_pos_list.union(features)
        
    # Save positional words to a single file
    with open(os.path.join(data_dir, 'fractal20220817_obj_data', '0.1.0', 'pos_vocab.json'), 'w') as f:
        json.dump(list(full_pos_list), f, indent=6)