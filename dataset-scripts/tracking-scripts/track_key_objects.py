import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import tensorflow_datasets as tfds
import time
import random
from dataclasses import dataclass
from typing import Any, List, Dict, Optional, Union, Tuple
import torch
import numpy as np
from PIL import Image
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from transformers import pipeline
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm
import argparse
import pickle
import os
# from openai import AzureOpenAI
import base64
import pdb
# ENDPOINT = "https://zhan-eastus-1.openai.azure.com"
# ORGANIZATION = "zhan-westus-0-global"
device = "cuda:1" if torch.cuda.is_available() else "cpu"
detector_id = "google/owlv2-large-patch14-ensemble"
# task_desc = ("You will be given an image with integer labeled bounding box proposals for the top, middle, and bottom drawers in a chest of drawers along with text specifying one of the three drawers. "
#              "Sometimes, the labels for all the drawers may not be present if they can't be seen in the image. If there is an open drawer in the image, then that is typically the drawer that is being referred to. Be aware that sometimes the open drawer may be cut off in the image or slightly occluded by an object. "
#              "If one of the labeled bounding boxes matches the specified drawer, answer with just the bounding box label like so: \"Correct: {box_index}\"."
#              " Otherwise, if the correct bounding box for the specified drawer is not present, "
#              "estimate the center of the drawer in normalized x & y coordinates from 0 to 1. "
#              " For this case, answer like so: \"Incorrect: {x_center},{y_center}\".")

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


def filter_detection_results(
    detection_results: List[DetectionResult],
    key_objects: List[str],
    position_list: List[str],
) -> List[Dict[str, Any]]:
    
    results_dict = {}
    for result in detection_results:
        
        results_dict[result.label] = results_dict.get(result.label, []) + [result]
            
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
        # label = "a "
        # if pos:
        #     label += f"{pos} "
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

def calculate_intersection_ratio(large_box, small_box):
    
    # Unpack coordinates
    x1_large, y1_large, x2_large, y2_large = large_box
    x1_small, y1_small, x2_small, y2_small = small_box
    
    # Calculate intersection coordinates
    x1_intersection = max(x1_large, x1_small)
    y1_intersection = max(y1_large, y1_small)
    x2_intersection = min(x2_large, x2_small)
    y2_intersection = min(y2_large, y2_small)
    
    # Calculate intersection area
    intersection_width = max(0, x2_intersection - x1_intersection)
    intersection_height = max(0, y2_intersection - y1_intersection)
    intersection_area = intersection_width * intersection_height
    
    # Calculate area of the small box
    small_box_area = (x2_small - x1_small) * (y2_small - y1_small)
    
    # Calculate the intersection ratio
    if small_box_area == 0:
        return 0  # To handle cases where the small box has zero area
    
    intersection_ratio = intersection_area / small_box_area
    
    return intersection_ratio

def call_chatgpt(chatgpt_messages, model="gpt-4o-mini-2024-07-18"):
    
    response = client.chat.completions.create(
        model=model, messages=chatgpt_messages
    )
    reply = response.choices[0].message.content
    print('*'*50)
    print(f'ChatGPT reply: {reply}')
    print('*'*50)
    
    if "Incorrect" in reply:
        return [float(x) for x in reply.split(":")[1].strip().split(",")]
    else:
        return None

    
def prepare_chatgpt_message(drawer, base64_image):
    messages = [{"role": "system", "content": task_desc}]
    messages += [
        {
            "role": "user",
            "content": [
                {'type': 'text', 'text': drawer},
                # {
                #     'type': 'image_url',
                #     'image_url':
                #     {
                #          'url': f"data:image/jpeg;base64,{example_img}"
                #     }
                # },
                {
                    'type': 'image_url',
                    'image_url':
                    {
                         'url': f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]
    return messages


def correct_drawer_bbox(drawer, img):
    
    requests = prepare_chatgpt_message(drawer, img)
    proposals = call_chatgpt(requests)
    
    return proposals


def drawer_detection(
    image: Image.Image,
    cabinet_box: List[int],
    drawer_position: str,
    width: int,
    task: str,
    threshold: float = 0.2,
) -> List:
    
    if 'close' in task or 'pick' in task or 'place' in task:
        object = ['open drawer']
    else:
        object = ['closed drawer']
    
    # Detect the drawer
    results = detect(
        image=image,
        key_objects=object,
        position_list=[drawer_position],
        threshold=threshold
    )
    results = [result.box.xyxy for result in results]
    
    # Remove detections that don't have a high IoU with the cabinet
    new_results = [
        result for result in results if calculate_intersection_ratio(cabinet_box, result) > 0.92
    ]
    
    if len(new_results) == 0:
        return [cabinet_box], cabinet_box
    
    # Remove detections that contain any other predictions inside of them
    filtered_results = []
    
    for i, result in enumerate(new_results):
        is_contained = False
        
        for other_result in results[:i] + results[i + 1:]:
            
            result_area = (result[2] - result[0]) * (result[3] - result[1])
            other_result_area = (other_result[2] - other_result[0]) * (other_result[3] - other_result[1])
            
            # Only check if it's a bigger box
            if result_area > other_result_area:
                if calculate_intersection_ratio(result, other_result) > 0.95:
                    is_contained = True
                    break
                
        if not is_contained:
            filtered_results.append(result)
    
    if len(filtered_results) == 0:
        return [cabinet_box], cabinet_box

    filtered_results = [result for result in filtered_results if (result[2] - result[0]) / width > 0.25]
    
    if len(filtered_results) == 0:
        return [cabinet_box], cabinet_box
    
    # Sort the results based on the y-coordinate
    filtered_results = sorted(filtered_results, key=lambda x: x[1])
    
    if drawer_position == 'top':
        return filtered_results, filtered_results[0]
    elif drawer_position == 'bottom':
        return filtered_results, filtered_results[-1]
    else:
        return filtered_results, filtered_results[len(filtered_results) // 2]

    
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
    
    # Sample image used for prompting 
    example_img = base64_image = encode_image('dataset-scripts/example_drawers.jpg')
    
    
    # client = AzureOpenAI(
    #     api_key=API_KEY,
    #     azure_endpoint=ENDPOINT,
    #     api_version="2023-05-15"
    # )
    
    object_detector = pipeline(model=detector_id, task="zero-shot-object-detection", device=device)
    # processor = AutoProcessor.from_pretrained(detector_id)
    # model = OwlViTForObjectDetection.from_pretrained(detector_id).to(device)
    predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large").to(device)
    
    shard_str_length = 5 - len(str(shard))
    shard_str = '0' * shard_str_length + str(shard)
    
    dataset = tfds.load('fractal20220817_data', data_dir=params.data_dir,
                        split=split)
    
    data_dict = {'idx': [idx for idx in range(len(dataset))], 
                 'timestep_length': [len(item['steps']) for item in dataset]}
    data_idx = tf.data.Dataset.from_tensor_slices(data_dict)
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

        if 'drawer' in key_objects:
            
            # Detect the whole cabinet first
            for i in range(len(key_objects)):
                key_objects[i] = key_objects[i].replace('drawer', 'chest of drawers')
            image_list = []
            
            vid_dir = 'vid_dir' if task else 'segment_dir'
            
            
            # Save all the image frames into the img_dir directory
            for frame in example['steps']:
                image = Image.fromarray(frame['observation']['image'].numpy())
                image_list.append(image)
                ts = frame['timestep']
                image.save(f"{vid_dir}/{ts}.jpg")
                
            
            # If this is a 'close' task, reverse the image list
            # if 'close' in task:
            #     rev_image_list = image_list[::-1]
        
            if task:
                
                # Detect objects from initial frame
                detected_objects = detect(image=Image.open('vid_dir/0.jpg'),
                                        key_objects=key_objects,
                                        position_list=position_list,
                                        threshold=0.18)
                detected_objects = filter_detection_results(detected_objects, key_objects, position_list)

                # Mask colors for each object 
                random_colors = np.random.randint(0, 255, size=(len(key_objects), 3), dtype=int)
                random_colors = [tuple(color) for color in random_colors]
                
                # Initialize inference state on this video and reset SAM
                inference_state = predictor.init_state(video_path='vid_dir')
                predictor.reset_state(inference_state)
                
                ann_obj_ids = list(range(len(key_objects)))
                
                for ann_obj_id, obj, pos in zip(ann_obj_ids, detected_objects, position_list):
                    
                    if obj != 'chest of drawers':
                    
                        # Instatiate masks for each object
                        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            inference_state=inference_state,
                            frame_idx=0,
                            obj_id=ann_obj_id,
                            box=detected_objects[obj],
                        )
                        
                    if obj == 'chest of drawers' and pos:
                        
                        end_idx = len(image_list) - 1 if 'close' in task and pos != 'bottom' else 1
                        
                        for frame_idx in range(0, end_idx):
                            detected_drawers, drawer = drawer_detection(image_list[frame_idx], detected_objects[obj], pos,
                                                                image_list[i].width, task)

                            if frame_idx == 0:
                                plt.close("all")
                                plt.imshow(image_list[0])
                                plt.axis('off')
                                
                                for i, d in enumerate(detected_drawers):
                                    show_box(d, plt.gca(), i)
                                
                                # Save drawer detection images
                                plt.savefig(f"segment_vid_dir/{task}_{example_idx}_{shard}.png", bbox_inches='tight', pad_inches=0)
                            
                            # Get GPT-4o proposal
                            # base64_image = encode_image(f"segment_vid_dir/{task}_{example_idx}.png")
                            # gpt_proposal = correct_drawer_bbox(f"{pos} drawer", base64_image)
                            
                            # If the drawer is not correctly detected, use GPt-4o point proposal
                            # if isinstance(gpt_proposal, list):
                            #     print('Using GPT-4o point proposal...')
                            #     print(f"segment_vid_dir/{task}_{example_idx}.png")
                            #     x_center, y_center = gpt_proposal
                            #     drawer_point = [x_center * image_list[0].width, y_center * image_list[0].height]
                            #     points = np.array([drawer_point], dtype=np.float32)
                            #     labels = np.array([1], dtype=np.int32)
                                
                            #     # Add the point to the drawer
                            #     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            #         inference_state=inference_state,
                            #         frame_idx=0,
                            #         obj_id=ann_obj_id,
                            #         points=points,
                            #         labels=labels,
                            #     )
                            #     show_points(points, labels, plt.gca(), marker_size=200)
                                
                            #     plt.savefig(f"segment_vid_dir/{task}_{example_idx}.png", bbox_inches='tight', pad_inches=0)
                            # # Otherwise choose the correct drawer
                            # else isinstance(gpt_proposal, int):
                            #     print('Using OWL-ViT box proposal...')
                            #     drawer = detected_drawers[gpt_proposal]
                                
                            #     # Readjust mask to specific drawer
                            #     _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                            #         inference_state=inference_state,
                            #         frame_idx=0,
                            #         obj_id=ann_obj_id,
                            #         box=drawer,
                            #     )
                            # else:
                            #     print('Using OWL-ViT box proposal...')
                                # Just default to drawer chosen by OWL-ViT
                            
                            
                            if frame_idx > 0:
                                
                                # Drawer should be moving slightly up if top drawer
                                if pos == 'top':
                                    if drawer[-1] > prev_drawer[-1]:
                                        continue
                                elif pos == 'middle':
                                    
                                    if (prev_diff < 0 and drawer[-1] > prev_drawer[-1]) or (prev_diff > 0 and drawer[-1] < prev_drawer[-1]):
                                        continue
                                    elif abs(drawer[-1] - prev_drawer[-1])/image_list[frame_idx].height > 0.05:
                                        continue
                                
                            prev_drawer = drawer
                            prev_diff = drawer[-1] - prev_drawer[-1]
                            
                            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=ann_obj_id,
                                box=drawer,
                            )
                            
                
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
        
            img_paths = [f"segment_dir/{img_name}" for img_name in os.listdir('segment_dir')]
            img_paths.sort()
            output_video = f"segment_vid_dir/{task}_{example_idx}_{shard}.gif"
            input_pattern = os.path.join('segment_dir', f'{task}_{example_idx}_%d_{shard}.png') if task else os.path.join('segment_dir', f'%d.jpg')
            os.system(f'ffmpeg -loglevel error -y -i \"{input_pattern}\"  \"{output_video}\"')
            os.system('rm -rf segment_dir/*')
            
            # Remove all the images from the vid_dir dierectory
            os.system('rm -rf vid_dir/*')

    print(f'Saving {img_idx} images to pickle file...')
    pickle_file = params.pickle_file_path
    with open(pickle_file, 'wb') as f:
        pickle.dump(images_data, f)
    print(f"Time taken: {time.time() - start_time}")
