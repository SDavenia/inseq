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


def extract_norms_backprop(model, img_pixels, extracted_grads):
    """
    img_pixels = inputs['pixel_values'] = inputs['pixel_values'].clone().detach().requires_grad_(True)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extracted_grads = extracted_grads.to(device)
    img_pixels = img_pixels.to(device)
    model = model.model
    model_vision_embeddings = model.vision_tower.vision_model.embeddings
    model_vision_encoder = model.vision_tower.vision_model.encoder
    model_vision_postlayernorm = model.vision_tower.vision_model.post_layernorm

    with torch.enable_grad():
        # Do them step by step
        model_vision_embeddings_output = model_vision_embeddings(img_pixels)
        model_vision_embeddings_output.retain_grad()  # Retain gradients for the non-leaf tensor
        model_vision_encoder_output = model_vision_encoder(model_vision_embeddings_output)
        model_vision_encoder_output_lhs = model_vision_encoder_output.last_hidden_state
        model_vision_encoder_output_lhs.retain_grad()
        model_postlayernorm_output = model_vision_postlayernorm(model_vision_encoder_output_lhs)
        model_postlayernorm_output.retain_grad()
        projected_image_features = model.multi_modal_projector(model_postlayernorm_output)
        projected_image_features.retain_grad()
    
    extracted_grads = extracted_grads.swapaxes(0, 1)
    extracted_grads = extracted_grads[:, :256, :]
    projected_image_features.backward(extracted_grads)
    input_gradients = model_vision_embeddings_output.grad
    input_gradients_norm = torch.norm(input_gradients, dim = 2).squeeze(0).cpu().numpy()
    return input_gradients_norm


def main():
    # Load the PaliGemma model and processor
    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
    # Prepare a sample image and text for testing (you can replace this with your own image/text)
    text = "Where is the cow standing"
    image = Image.new('RGB', (224, 224), (0, 0, 0))  
    inputs = processor(text=[text], images=image, return_tensors="pt")
    inputs['pixel_values'] = inputs['pixel_values'].clone().detach().requires_grad_(True)
    extracted_grads = torch.load('../tensor_trial.pt')
    input_gradients_norm = extract_norms_backprop(model, processor, inputs['pixel_values'], extracted_grads)

    print(f"Input_gradients_norm shape: {input_gradients_norm.shape}")
    print(f"Input_gradients_norm: {input_gradients_norm}")

if __name__ == '__main__':
    main()