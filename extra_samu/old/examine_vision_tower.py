import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

# Load the PaliGemma model and processor
model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224")
processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")
print(f"\nVision tower:")
print(model.vision_tower)
print(f"---------------------------------------------------")
model_vision_embeddings = model.vision_tower.vision_model.embeddings
print(f"Model vision embeddings:\n{model_vision_embeddings}")
print(f"---------------------------------------------------")
model_vision_encoder = model.vision_tower.vision_model.encoder
print(f"Model vision encoder: {model_vision_encoder}")
print(f"---------------------------------------------------")
model_vision_postlayernorm = model.vision_tower.vision_model.post_layernorm
print(f"Model vision postlayernorm: {model_vision_postlayernorm}")
print(f"---------------------------------------------------")
print(f"\n\n")

# Prepare a sample image and text for testing (you can replace this with your own image/text)
image = Image.new('RGB', (224, 224), (0, 0, 0))  
text = "Where is the cow standing?"

# Tokenize the input (image and text)
inputs = processor(text=[text], images=image, return_tensors="pt")
inputs['pixel_values'] = torch.tensor(inputs['pixel_values'], requires_grad=True)

# Step by step vision tower.
model_vision_embeddings_output = model_vision_embeddings(inputs['pixel_values'])
print(f"Model vision embeddings output shape: {model_vision_embeddings_output.shape}")
model_vision_encoder_output = model_vision_encoder(model_vision_embeddings_output)
model_vision_encoder_output_lhs = model_vision_encoder_output.last_hidden_state
print(f"Model vision encoder output shape: {model_vision_encoder_output_lhs.shape}")
model_postlayernorm_output = model_vision_postlayernorm(model_vision_encoder_output_lhs)
print(f"Model postlayernorm output shape: {model_postlayernorm_output.shape}")

print(f"---------------------------------------------------")
print(f"Check that outputs are the same")
print(f"Original vision output:")
vision_output = model.vision_tower(inputs['pixel_values'])
vision_output_lhs = vision_output.last_hidden_state
print(vision_output_lhs)
print(f"Extracted vision_output")
print(model_postlayernorm_output)

