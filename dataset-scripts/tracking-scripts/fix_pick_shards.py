import os, json, argparse


if __name__ == '__main__':
    
    args = argparse.ArgumentParser(description='Fix pick shards')
    args.add_argument('--data-dir', type=str, default='/ariesdv0/zhanling/oxe-data-converted')
    args = args.parse_args()
    shard_lengths = []
    
    for i in range(1025):
        with open(os.path.join(args.data_dir, 'fractal20220817_pick_data/0.1.0', f'num_examples_{i}.json'), 'r') as f:
            
            # Load the shard length and append it to the list
            num_examples = json.load(f)[0]
            shard_lengths.append(str(num_examples))
            os.remove(os.path.join(args.data_dir, 'fractal20220817_pick_data/0.1.0', f'num_examples_{i}.json'))
    
    dset_info_path = os.path.join(
        args.data_dir,
        'fractal20220817_obj_data',
        '0.1.0',
        'dataset_info.json'
    )
    
    with open(dset_info_path, 'r') as f:
        dset_info = json.load(f)
    
    # Update the shard lengths
    dset_info["splits"][0]["shardLengths"] = shard_lengths
    
    with open(os.path.join(args.data_dir, 'fractal20220817_pick_data/0.1.0', 'dataset_info.json'), 'w') as f:
        json.dump(dset_info, f, indent=6)