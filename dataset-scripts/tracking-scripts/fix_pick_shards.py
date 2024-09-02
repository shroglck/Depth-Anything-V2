import os, json, argparse
import tensorflow_datasets as tfds
from tqdm import tqdm


if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Fix pick shards')
    args.add_argument('--data-dir', type=str, default='/ariesdv0/zhanling/oxe-data-converted')
    args = args.parse_args()
    shard_lengths = []
    
    dset_info_path = os.path.join(
        args.data_dir,
        'fractal20220817_pick_data',
        '0.1.0',
        'dataset_info.json'
    )
    with open(dset_info_path, 'r') as f:
        dset_info = json.load(f)
    print('Starting to fix pick shards...')
    
    for shard in tqdm(range(1024)):
        
        split = f'train[{shard}shard]'
        print(f'Getting # of examples in {split}')
    
        shard_str_length = 5 - len(str(shard))
        shard_str = '0' * shard_str_length + str(shard)
        
        # Load pickle file and dataset/record dataset
        dataset = tfds.load('fractal20220817_pick_data', data_dir=args.data_dir,
                            split=split)

        num_examples = 0
        
        # Get the number of examples in the shard
        for example in dataset:
            
            task = [e['observation']['natural_language_instruction'] for e in example['steps'].take(1)][0]
            
            if task:
                num_examples += 1
        
        
        dset_info["splits"][0]["shardLengths"][shard] = str(num_examples)
    
    
    dset_info['name'] = 'fractal20220817_seg_data'
    with open(os.path.join(args.data_dir, 'fractal20220817_seg_data', '0.1.0', 'dataset_info.json'), 'w') as f:
        json.dump(dset_info, f, indent=6)
        