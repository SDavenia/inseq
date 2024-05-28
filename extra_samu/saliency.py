"""
From the original paper proposing the Contrastive attribution framework.
"""
import argparse, json
import random
import torch
import numpy as np
from transformers import (
    WEIGHTS_NAME,
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPTNeoForCausalLM,

)

import matplotlib as mpl
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 10]

config = GPT2Config.from_pretrained("gpt2")
VOCAB_SIZE = config.vocab_size

# Adapted from AllenNLP Interpret and Han et al. 2020
def register_embedding_list_hook(model, embeddings_list):
    def forward_hook(module, inputs, output):
        embeddings_list.append(output.squeeze(0).clone().cpu().detach().numpy())
    embedding_layer = model.transformer.wte
    handle = embedding_layer.register_forward_hook(forward_hook)
    return handle

def register_embedding_gradient_hooks(model, embeddings_gradients):
    def hook_layers(module, grad_in, grad_out):
        embeddings_gradients.append(grad_out[0].detach().cpu().numpy())
    embedding_layer = model.transformer.wte
    hook = embedding_layer.register_backward_hook(hook_layers)
    return hook

def saliency(model, input_ids, input_mask, batch=0, correct=None, foil=None):
    # Get model gradients and input embeddings
    torch.enable_grad()
    model.eval()
    embeddings_list = []
    handle = register_embedding_list_hook(model, embeddings_list)
    embeddings_gradients = []
    hook = register_embedding_gradient_hooks(model, embeddings_gradients)
    
    if correct is None:
        correct = input_ids[-1]
    input_ids = input_ids[:-1]
    input_mask = input_mask[:-1]
    input_ids = torch.tensor(input_ids, dtype=torch.long).to(model.device)
    input_mask = torch.tensor(input_mask, dtype=torch.long).to(model.device)

    model.zero_grad()
    A = model(input_ids, attention_mask=input_mask)

    if foil is not None and correct != foil:
        (A.logits[-1][correct]-A.logits[-1][foil]).backward()
    else:
        (A.logits[-1][correct]).backward()
    handle.remove()
    hook.remove()

    return np.array(embeddings_gradients).squeeze(), np.array(embeddings_list).squeeze()

def l1_grad_norm(grads, normalize=False):
    l1_grad = np.linalg.norm(grads, ord=1, axis=-1).squeeze()

    if normalize:
        norm = np.linalg.norm(l1_grad, ord=1)
        l1_grad /= norm
    return l1_grad

def visualize(attention, tokenizer, input_ids, gold=None, normalize=False, print_text=True, save_file=None, title=None, figsize=60, fontsize=36):
    tokens = [tokenizer.decode(i) for i in input_ids[0][:len(attention) + 1]]
    if gold is not None:
        for i, g in enumerate(gold):
            if g == 1:
                tokens[i] = "**" + tokens[i] + "**"

    # Normalize to [-1, 1]
    if normalize:
        a,b = min(attention), max(attention)
        x = 2/(b-a)
        y = 1-b*x
        attention = [g*x + y for g in attention]
    attention = np.array([list(map(float, attention))])

    fig, ax = plt.subplots(figsize=(figsize,figsize))
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    im = ax.imshow(attention, cmap='seismic', norm=norm)

    if print_text:
        ax.set_xticks(np.arange(len(tokens)))
        ax.set_xticklabels(tokens, fontsize=fontsize)
    else:
        ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    for (i, j), z in np.ndenumerate(attention):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center', fontsize=fontsize)


    ax.set_title("")
    fig.tight_layout()
    if title is not None:
        plt.title(title, fontsize=36)
    
    if save_file is not None:
        plt.savefig(save_file, bbox_inches = 'tight',
        pad_inches = 0)
        plt.close()
    else:
        plt.show()

def main():
    # Define model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare input:
    input = "Can you stop the dog from" 
    input = input.strip() + " "
    input_tokens = tokenizer(input)['input_ids']
    attention_ids = tokenizer(input)['attention_mask']

    # Get saliency:
    target = "barking" 
    foil = "fighting" 
    explanation = "gradient norm" 
    CORRECT_ID = tokenizer(" "+ target)['input_ids'][0]
    FOIL_ID = tokenizer(" "+ foil)['input_ids'][0]

    
    target_saliency_matrix, target_embd_matrix = saliency(model, input_tokens, attention_ids)               # Backprop w.r.t. prob(correct)
    foil_saliency_matrix, foil_embd_matrix = saliency(model, input_tokens, attention_ids, correct=FOIL_ID)  # Backpropr w.r.t. prob(foil)

    saliency_matrix, embd_matrix = saliency(model, input_tokens, attention_ids, foil=FOIL_ID) # Backpropr w.r.t. prob(correct) - prob(foil)

    print(f"Saliency matrix: {saliency_matrix[0:2, 0:5]}")
    print(f"Target saliency matrix: {(target_saliency_matrix-foil_saliency_matrix)[0:2, 0:5]}") # THIS IS CORRECT -> SO DUNNO WHAT THEY ARE DOING IN INSEQ!!!
 


if __name__ == "__main__":
    main()