import numpy as np
import pandas as pd
import subprocess
import tempfile
import os
from PIL import Image
import PIL
"""
export MKL_SERVICE_FORCE_INTEL=1

IGNORE GO TO THE ONE IN VLM_EXP
"""

def save_image_to_temp(image, index):
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    
    # Define the path where the image will be saved
    image_path = os.path.join(temp_dir, f"image_{index}.jpg")
    
    # Save the PIL image to the temporary path
    image.save(image_path)
    return image_path

def process_entry(row, index):
    # Convert PIL image to a temporary file path
    image_path = save_image_to_temp(row['image'], index)
    
    # Extract item and prepare input text
    item = row['item']
    print(row)
    input_text = f"What is the price of {item}?"
    print(f"input text: {input_text}")
    print(f"Correct answer: {row['price']}")
    
    # Run the inseq command with the temporary image path
    run_inseq_command(image_path, input_text)

def run_inseq_command(image_path, input_current_text):
    command = [
        "python", 
        "-m", 
        "inseq.commands.cli", 
        "attribute-context", 
        "--input_current_text", 
        input_current_text, 
        "--attributed_fn", 
        "contrast_prob_diff", 
        "--model_name", 
        "google/paligemma-3b-mix-224", 
        "--generation_kwargs", 
        '{"max_new_tokens": 50}', 
        "--context_image_path", 
        image_path
        # TODO: Add command --return_cti --return bboxes
    ]
    
    result = subprocess.run(command, capture_output=True, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return Code:", result.returncode)

import os

def extract_cti_bboxes(save_path):
    # Directory path containing the bboxes files
    bboxes_path = os.path.join(save_path, 'bboxes')
    cci_scores_path = os.path.join(save_path, 'cci_scores')

    # Lists to store words and bounding boxes
    words_list = []
    bboxes_list = []
    cci_scores_list = []

    # Loop through all files in the directory
    for file_name in os.listdir(bboxes_path):
        if file_name.endswith('_bboxes.txt'):
            # Extract the word from the filename
            word = file_name.split('_bboxes.txt')[0]
            words_list.append(word)

            # Full path to the file
            bboxes_file_path = os.path.join(bboxes_path, file_name)

            # List to store bounding boxes for the current file
            current_bboxes = []

            # Open and read the file content
            with open(bboxes_file_path, 'r') as file:
                for line in file:
                    # Convert the bounding box string into a list of integers
                    bbox = list(map(int, line.strip().split(',')))
                    current_bboxes.append(bbox)

            # Append the list of bounding boxes for this word to bboxes_list
            bboxes_list.append(current_bboxes)

            # Delete the file after processing
            os.remove(bboxes_file_path)
    
    # Repeat for cci_scores
    for file_name in os.listdir(cci_scores_path):
        if file_name.endswith('cci_scores.txt'):
            cci_scores_file_path = os.path.join(cci_scores_path, file_name)

            # List to store bounding boxes for the current file
            current_cci_scores = []

            # Open and read the file content
            with open(cci_scores_file_path, 'r') as file:
                for line in file:
                    # Convert the bounding box string into a list of integers
                    score = line.strip()
                    current_cci_scores.append(score)

            # Append the list of bounding boxes for this word to bboxes_list
            cci_scores_list.append(current_cci_scores)

            # Delete the file after processing
            os.remove(cci_scores_file_path)


    return words_list, bboxes_list, cci_scores_list


if __name__ == "__main__":
    # Load the large dataframe from pkl using dask
    file_path = "/u/sdavenia/VLM_Experiments/datasets_exploration/correct-paligemma-3b-mix-224.pkl"
    df = pd.read_pickle(file_path).reset_index()

    save_path = '/u/sdavenia/inseq/extra_samu/wildreceipts_results'

    # Iterate over each row, processing one entry at a time
    all_cti_tokens_list = []
    all_bboxes_list = []
    all_cci_scores_list = []
    for idx, row in df.iterrows():
        # Check that directories are empty as they should be emptied after each call:
        if os.listdir(f"{save_path}/bboxes") or os.listdir(f"{save_path}/cci_scores"):  # os.listdir() returns a list of items in the directory
            raise RuntimeError(f"The directory is not empty.")
        
        process_entry(row, idx)
        cti_tokens_list, bboxes_list, cci_scores_list = extract_cti_bboxes(save_path)
        all_cti_tokens_list.append(cti_tokens_list)
        all_bboxes_list.append(bboxes_list)
        all_cci_scores_list.append(cci_scores_list)

    
    df['cti_tokens'] = all_cti_tokens_list
    df['bboxes'] = all_bboxes_list
    df['cci_scores'] = all_cci_scores_list
    final_df_path = '/u/sdavenia/inseq/extra_samu/final_df.pkl'
    df.to_pickle(final_df_path)

        
