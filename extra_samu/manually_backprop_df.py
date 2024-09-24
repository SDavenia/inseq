import torch
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'VLM_Experiments', 'wildreceipts_evaluation')))
from evaluate_pecore_utils import list_all_rectangles
from extract_viz_utils import plot_box_dict

def main():
    # Load the PaliGemma model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    model_vision_embeddings = model.vision_tower.vision_model.embeddings
    model_vision_encoder = model.vision_tower.vision_model.encoder
    model_vision_postlayernorm = model.vision_tower.vision_model.post_layernorm

    # Prepare a sample image and text for testing (you can replace this with your own image/text)
    df = pd.read_pickle('temp.pkl').reset_index(drop=True)
    row = df.iloc[0]

    # Tokenize the input (image and text)
    text = f"What is the price of {row['item']}"
    inputs = processor(text=[text], images=row['image'], return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].clone().detach().requires_grad_(True)
        
    input_gradients_norm = row['cci_scores'][1]
    backpropagated_gradients_norm = row['backprop_cci_scores'][1]

    resized_image = row['image'].resize((224, 224))
    resized_image.save('resized_image_df.jpg')
    patch_shape = (14, 14)
    n_rect = (16, 16)
    all_rect = list_all_rectangles(patch_shape, n_rect)

    # Original gradients.
    all_rect_original_grad_dict = {el: 0 for el in all_rect}
    for idx, rect in enumerate(all_rect):
        all_rect_original_grad_dict[rect] = float(input_gradients_norm[idx])

    plot_box_dict(box_dict=all_rect_original_grad_dict,
                model_img_shape=(224, 224),
                patch_shape=(14, 14),
                save_to='input_gradients_norm.png')

    # Backpropagated gradients.
    all_rect_backprop_grad_dict = {el: 0 for el in all_rect}
    for idx, rect in enumerate(all_rect):
        all_rect_backprop_grad_dict[rect] = float(backpropagated_gradients_norm[idx])

    plot_box_dict(box_dict=all_rect_backprop_grad_dict,
                model_img_shape=(224, 224),
                patch_shape=(14, 14),
                save_to='backpropagated_gradients_norm.png')

    
   
if __name__ == '__main__':
    main()