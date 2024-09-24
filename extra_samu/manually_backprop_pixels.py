"""
Try to backpropagate back to the pixels.
"""
import torch
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
from torchviz import make_dot
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'VLM_Experiments', 'wildreceipts_evaluation')))
from evaluate_pecore_utils import list_all_rectangles
from extract_viz_utils import plot_box_dict

def main():
    # Load the PaliGemma model and processor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extra_path = 'pixel_extractions'
    model = PaliGemmaForConditionalGeneration.from_pretrained("google/paligemma-3b-mix-224").to(device)
    processor = AutoProcessor.from_pretrained("google/paligemma-3b-mix-224")

    df = pd.read_pickle('../../VLM_Experiments/wildreceipts_evaluation/wildreceipts_correct_paligemma-3b-mix-224.pkl').reset_index(drop=True)
    row = df.iloc[10]

    text = f"What is the price of {row['item']}?"
    print(text)
    img = row['image']
    # img = Image.new('RGB', (100, 100), color=(0, 0, 0))
    # Tokenize the input (image and text)
    inputs = processor(text=[text], images=img, return_tensors="pt").to(device)
    input_pixel_values = inputs['pixel_values']
    input_pixel_values = input_pixel_values.requires_grad_(True)
    print(input_pixel_values)
    # inputs['pixel_values'] = inputs['pixel_values'].clone().detach().requires_grad_(True)
    # inputs['input_ids'] = inputs['input_ids'].clone().detach().requires_grad_(True)
    with torch.enable_grad():
        vision_tower_output = model.vision_tower(input_pixel_values).last_hidden_state
        vision_tower_output.retain_grad()
        output = model.multi_modal_projector(vision_tower_output)
    
    print(output.shape) # Tensor of shape [1, 256, 2048]

    extracted_grads = torch.load('/u/sdavenia/VLM_Experiments/wildreceipts_evaluation/gradients/step_0.pt').to(device)
    extracted_grads = extracted_grads.swapaxes(0, 1)
    #extracted_grads = extracted_grads[:, :-1, :]
    extracted_grads = extracted_grads[:, :256, :]


    output.backward(extracted_grads)
    # Extract the input gradients with respect to the pixel values (image)
    vision_tower_output_gradients = vision_tower_output.grad
    input_pixel_values_gradients = input_pixel_values.grad

    print(f"Shape of vision_tower_output_gradients: {vision_tower_output_gradients.shape}") # [1, 256, 1152]  
    print(f"Shape of input pixel values: {input_pixel_values_gradients.shape}")             # [1, 3, 224, 224]

    input_pixel_values_norm = torch.norm(input_pixel_values_gradients, dim = 1).squeeze(0).cpu().detach().numpy()
    print(input_pixel_values_norm.shape)
    
    patch_shape = (14, 14)
    n_rect = (16, 16)
    all_rect = list_all_rectangles(patch_shape, n_rect)

    # Gradient attribution scores using inseq extracted gradients (i.e. those w.r.t. the patches given as input to the LLM).
    extracted_grads_norm = torch.norm(extracted_grads, dim=2).squeeze(0).cpu().numpy()
    all_rect_original_grad_dict = {el: 0 for el in all_rect}
    for idx, rect in enumerate(all_rect):
        all_rect_original_grad_dict[rect] = float(extracted_grads_norm[idx])
    plot_box_dict(box_dict=all_rect_original_grad_dict,
                model_img_shape=(224, 224),
                patch_shape=(14, 14),
                save_to=f'{extra_path}/extracted_grads_norms.png')

    # Gradient attribution scores using vision tower output gradients.
    visiontower_grads_norm = torch.norm(vision_tower_output_gradients, dim=2).squeeze(0).cpu().numpy()
    all_rect_visiontower_grad_dict = {el: 0 for el in all_rect}
    for idx, rect in enumerate(all_rect):
        all_rect_visiontower_grad_dict[rect] = float(visiontower_grads_norm[idx])
    plot_box_dict(box_dict=all_rect_visiontower_grad_dict,
                model_img_shape=(224, 224),
                patch_shape=(14, 14),
                save_to=f'{extra_path}/visiontower_grads_norms.png')

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    # Create the plot
    plt.imshow(input_pixel_values_norm)
    plt.colorbar()
    plt.savefig(f"{extra_path}/input_pixels_norm.png")

    """
    import numpy as np
    from PIL import Image, ImageDraw

    def overlay_grid(image, mask_array):
        # Load the image and reshape to 224 x 224
        image_array = np.array(image)

        # Create a draw object
        draw = ImageDraw.Draw(image)

        # Set grid size and box size
        grid_size = 14
        box_size = 224 // grid_size

        # Check that mask_array has 256 entries
        if len(mask_array) != 256:
            raise ValueError("mask_array must contain exactly 256 entries.")

        # Iterate over the grid from top right to bottom left
        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate the grid box position
                box_x1 = (grid_size - 1 - j) * box_size
                box_y1 = i * box_size
                box_x2 = box_x1 + box_size
                box_y2 = box_y1 + box_size
                
                # Get the corresponding entry in mask_array
                mask_value = mask_array[i * grid_size + j]
                
                # Check the mask value
                if mask_value == 1:
                    # Draw a red rectangle
                    draw.rectangle([box_x1, box_y1, box_x2, box_y2], fill=(255, 0, 0, 128))

        # Save or show the modified image
        image.save('output_image.png')

    # Example usage
    # mask_array should be a list with 256 entries (0 or 1)
    mask_array = modified_gradients
    overlay_grid(img, mask_array)
    """

if __name__ == '__main__':
    main()



"""
import networkx as nx
def build_graph(fn, graph=None, visited=None):

    Recursively builds the computational graph.

    Args:
        fn: The starting grad_fn or backward function.
        graph: A networkx DiGraph object to store the graph.
        visited: A set to store the visited nodes and avoid loops.
    Returns:
        graph: The updated graph.

    if graph is None:
        graph = nx.DiGraph()
    if visited is None:
        visited = set()

    # If fn is None, return
    if fn is None:
        return graph
    
    # Get the name of the current node
    node_name = str(type(fn).__name__)

    # If we've already visited this node, return to avoid loops
    if node_name in visited:
        return graph

    # Mark this node as visited
    visited.add(node_name)
    graph.add_node(node_name)
    
    # Iterate through next_functions, which are the inputs to this backward function
    if hasattr(fn, 'next_functions'):
        for next_func in fn.next_functions:
            if next_func[0] is not None:
                next_node_name = str(type(next_func[0]).__name__)
                graph.add_edge(node_name, next_node_name)
                # Recursively add the next function to the graph
                build_graph(next_func[0], graph, visited)
    
    return graph

# Start building the graph from the grad_fn of the final output tensor
graph = build_graph(merge_output_tensor.grad_fn)

# Visualize or use the graph (for example, using networkx and matplotlib)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
nx.draw(graph, with_labels=True, node_color='lightblue', font_weight='bold', node_size=1500)
plt.savefig('graph.png')
"""

