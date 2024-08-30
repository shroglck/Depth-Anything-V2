import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import time
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
from torchvision.ops import nms
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm
import argparse
import pickle
import os
import base64
import pdb
device = "cuda" if torch.cuda.is_available() else "cpu"
detector_id = "google/owlv2-large-patch14-ensemble"

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"
from plot_utils import *

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def show_box(box, ax, box_number):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, str(box_number), color='white', fontsize=12, bbox=dict(facecolor='green', alpha=0.5, edgecolor='none'))
    

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    

def show_mask(mask, ax, obj_id=None, random_color=False, save_path=None):

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou

def filter_detection_results(
    detection_results: List[DetectionResult],
    key_objects: List[str],
    position_list: List[str],
) -> List[Dict[str, Any]]:
    
    boxes = torch.tensor([result.box.xyxy for result in detection_results]).to(torch.float32)
    scores = torch.tensor([result.score for result in detection_results])
    filtered_boxes = nms(boxes, scores, iou_threshold=0.5)
    filtered_detection_results = [detection_results[idx.item()] for idx in filtered_boxes]
    
    results_dict = {}
    for result in filtered_detection_results:
        results_dict[result.label] = results_dict.get(result.label, []) + [result]
    
    # Get the most confident prediction for each object
    results_dict = {obj: sorted(results_dict[obj], key=lambda x: x.score, reverse=True) for obj in key_objects if obj in results_dict}
    results_dict = {obj: results_dict[obj][:1] for obj in key_objects if obj in results_dict}
    
    for key_object in key_objects:
        if key_object not in results_dict:
            
            # Sort the predictions based on the score
            predictions = sorted([result for result in detection_results if result.label == key_object], key=lambda x: x.score, reverse=True)
            
            # If the best prediction doesn't overlap with any of the existing predictions, add it
            for prediction in predictions:
                if all(bb_intersection_over_union(prediction.box.xyxy, result[0].box.xyxy) < 0.5 for result in results_dict.values()):
                    results_dict[key_object] = [prediction]
                    break
            else:
                results_dict[key_object] = [predictions[0]]
    
    # for object in results_dict:
    #     for result in results_dict[object]:
    #         show_box(result.box.xyxy, plt.gca(), f'{result.label} {result.score:.2f}')
        
    results_dict = {obj: sorted(results_dict[obj], key=lambda x: x.score, reverse=True) for obj in key_objects if obj in results_dict} 
    final_results = {}
    
    for position, key_object in zip(position_list, key_objects):
    
        # If there is no associated position, just use the most confident prediction
        if not position:
            best_prediction = results_dict[key_object][0].box.xyxy
            final_results[key_object] = best_prediction
        else:
            prediction_length = len(results_dict[key_object])
            
            # Sort the predictions based on the position
            if position in ['top', 'bottom', 'middle', 'front', 'back']:
                results_dict[key_object] = sorted(results_dict[key_object], key=lambda x: x.box.ymin)
                if position in ['top', 'front']:
                    final_results[key_object] = results_dict[key_object][0].box.xyxy
                elif position in ['back', 'bottom']:
                    final_results[key_object] = results_dict[key_object][-1].box.xyxy
                else:
                    final_results[key_object] = results_dict[key_object][prediction_length // 2].box.xyxy
            elif position in ['left', 'right']:
                results_dict[key_object] = sorted(results_dict[key_object], key=lambda x: x.box.xmin)
                if position == 'left':
                    final_results[key_object] = results_dict[key_object][0].box.xyxy
                else:
                    final_results[key_object] = results_dict[key_object][-1].box.xyxy
    
    return final_results
        

def detect(
    image: Image.Image,
    key_objects: List[str],
    position_list: List[str],
    threshold: float = 0.15,
) -> List[Dict[str, Any]]:
    
    """
    Use Grounding DINO to detect a set of labels in an image in a zero-shot fashion.
    """
    
    labels = []
    
    for obj, pos in zip(key_objects, position_list):
        label = "a photo of a "
        label += f"{obj}"
        labels.append(label)
    
    label_dict = {label: key_object for label, key_object in zip(labels, key_objects)}
    detector_input = [
        {
            'image': image,
            'candidate_labels': labels
        }
    ]
        
    finished_detection = False
    
    while not finished_detection:
    
        # Keep on making predictions until all objects are detected
        results = object_detector(detector_input, threshold=threshold)
        
        if results:
            
            results = results[0]
            
            # Change label 
            for i in range(len(results)):
                results[i]['label'] = label_dict[results[i]['label']]
            
            finished_detection = all([key_object in set([pred['label'] for pred in results]) for key_object in key_objects])
            
        if not finished_detection:
            threshold -= 0.01
            
    results = [DetectionResult.from_dict(pred) for pred in results]
    
    return results

    
def add_timestep_index(example, index):
    
    def add_step_index(example, index):
        example['timestep'] = index
        return example
    
    timestep_index = tf.range(index['timestep_length'])
    timestep_index = tf.data.Dataset.from_tensor_slices(timestep_index)
    example['steps'] = tf.data.Dataset.zip((example['steps'], timestep_index))
    example['steps'] = example['steps'].map(add_step_index)
    example['idx'] = index['idx']
    
    return example

def params():
    
    parser = argparse.ArgumentParser(description='Save dataset with depth images')
    parser.add_argument('--data-shard', type=int, default=0,
                        help='Shard of the dataset to save', choices=[i for i in range(1024)])
    parser.add_argument('--data-dir', type=str, default='/data/shresth/octo-data')
    parser.add_argument('--pickle_file_path', type=str, default='segment_images.pkl')
    args = parser.parse_args()
    return args


if __name__ == '__main__':   
    
    params = params()
    shard = params.data_shard
    split = f'train[{shard}shard]'
    
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to(device)
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    dataset = tfds.load('fractal20220817_pick_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))], 
                 'timestep_length': [len(item['steps']) for item in dataset]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
    dataset = tf.data.Dataset.zip((dataset, data_idx))
    dataset = dataset.map(add_timestep_index, num_parallel_calls=1)
    
    image_batch, object_list, instruction_batch = [], [], []
    object_dict = {}
    
    pos_vocab = set()
    
    print(f'Starting to segment key objects for shard {shard}...')
    start_time = time.time()
    os.makedirs('vid_dir', exist_ok=True)
    os.makedirs('segment_dir', exist_ok=True)
    
    img_idx = 0
    images_data = {}

    # for example in dataset:
    for i, example in tqdm(enumerate(dataset), total=len(dataset)):
        
        task = [d['observation']['natural_language_instruction'].numpy().decode('utf-8') for d in example['steps'].take(1)][0]
        example_idx = int(example['idx'].numpy())
        key_objects = [object.numpy().decode('utf-8') for object in example['key_objects']]
        position_list = [position.numpy().decode('utf-8') for position in example['positional_words']]

        image_list = []
        
        vid_dir = 'vid_dir' if task else 'segment_dir'
        
        
        # Save all the image frames into the img_dir directory
        for frame in example['steps']:
            image = Image.fromarray(frame['observation']['image'].numpy())
            image_list.append(image)
            ts = frame['timestep']
            image.save(f"{vid_dir}/{ts}.jpg")

        if task:
            
            # plt.close("all")
            # plt.imshow(image_list[0])
            # plt.axis('off')
            
            # Detect objects from initial frame
            detected_objects = detect(image=Image.open('vid_dir/0.jpg'),
                                    key_objects=key_objects,
                                    position_list=position_list,
                                    threshold=0.3)
            detected_objects = filter_detection_results(detected_objects, key_objects, position_list)

            # Initialize inference state on this video and reset SAM
            inference_state = predictor.init_state(video_path='vid_dir')
            predictor.reset_state(inference_state)
            
            ann_obj_ids = list(range(len(key_objects)))
            
            for ann_obj_id, obj, pos in zip(ann_obj_ids, detected_objects, position_list):
                
                # Instatiate masks for each object
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    box=detected_objects[obj],
                )
                # show_box(detected_objects[obj], plt.gca(), i)
                    
            # Save drawer detection images
            # plt.savefig(f"segment_vid_dir/{task}_{example_idx}_{shard}.png", bbox_inches='tight', pad_inches=0)
                        
            video_segments = {}
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
            
            for image, mask, ts in zip(image_list, video_segments, range(len(video_segments))):
                img_name = f"{task}_{example_idx}_{ts}_{shard}.png"
                
                plt.close("all")
                plt.imshow(image)
                plt.axis('off')
                for out_obj_id, out_mask in video_segments[ts].items():
                    show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

                plt.savefig(f"segment_dir/{img_name}", bbox_inches='tight', pad_inches=0)
                seg_img = Image.open(f"segment_dir/{img_name}")
                images_data[img_name] = seg_img
                img_idx += 1
        else:
            for img, ts in zip(image_list, range(len(image_list))):

                img_name = f"{task}_{example_idx}_{shard}.png"
                seg_img = Image.open(f"segment_dir/{ts}.jpg")
                images_data[img_name] = seg_img
                img_idx += 1

        # img_paths = [f"segment_dir/{img_name}" for img_name in os.listdir('segment_dir')]
        # img_paths.sort()
        # output_video = f"segment_vid_dir/{task}_{example_idx}_{shard}.gif"
        # input_pattern = os.path.join('segment_dir', f'{task}_{example_idx}_%d_{shard}.png') if task else os.path.join('segment_dir', f'%d.jpg')
        # os.system(f'ffmpeg -loglevel error -y -i \"{input_pattern}\"  \"{output_video}\"')
        os.system('rm -rf segment_dir/*')
        print(f'Saving video for {output_video}...')
        
        # Remove all the images from the vid_dir dierectory
        os.system('rm -rf vid_dir/*')
           
    print(f'Saving {img_idx} images to pickle file...')
    pickle_file = params.pickle_file_path
    with open(pickle_file, 'wb') as f:
        pickle.dump(images_data, f)
    print(f"Time taken: {time.time() - start_time}")
