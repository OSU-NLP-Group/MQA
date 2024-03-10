import os
import random
from PIL import Image
import torch


def call_blip2_engine_df(sample, model):
    prompt = sample['final_input_prompt']
    image = sample['image']
    response = model.generate({"image": image, "prompt": prompt}, max_length=5)[0]
    return response

def blip_image_processor(img_path, vis_processors):
    try:
        raw_image = Image.open(img_path).convert('RGB')
    except:
        img_path = os.path.dirname(img_path)
        img_path = os.path.join(img_path, 'inf.png')
        raw_image = Image.open(img_path).convert('RGB')
    image = vis_processors["eval"](raw_image).unsqueeze(0)
    return image

