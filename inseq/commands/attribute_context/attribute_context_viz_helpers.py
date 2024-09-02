from copy import deepcopy
from typing import Literal, Optional, Union

from rich.console import Console

from ... import load_model
from ...models import HuggingfaceModel
from .attribute_context_args import AttributeContextArgs
from .attribute_context_helpers import (
    AttributeContextOutput,
    filter_rank_tokens,
    get_filtered_tokens,
    get_scores_threshold,
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def get_formatted_procedure_details(args: AttributeContextArgs) -> str:
    def format_comment(std: Optional[float] = None, topk: Optional[int] = None) -> str:
        comment = []
        if std:
            comment.append(f"std λ={std:.2f}")
        if topk:
            comment.append(f"top {topk}")
        if len(comment) > 0:
            return ", ".join(comment)
        return "all"

    cti_comment = format_comment(args.context_sensitivity_std_threshold, args.context_sensitivity_topk)
    cci_comment = format_comment(args.attribution_std_threshold, args.attribution_topk)
    input_context_comment, output_context_comment = "", ""
    if args.has_input_context:
        input_context_comment = f"\n[bold]Input context:[/bold]\t{args.input_context_text}"
    if args.has_output_context:
        output_context_comment = f"\n[bold]Output context:[/bold]\t{args.output_context_text}"
    return (
        f"\nContext with [bold green]contextual cues[/bold green] ({cci_comment}) followed by output"
        f" sentence with [bold dodger_blue1]context-sensitive target spans[/bold dodger_blue1] ({cti_comment})\n"
        f'(CTI = "{args.context_sensitivity_metric}", CCI = "{args.attribution_method}" w/ "{args.attributed_fn}" '
        f"target)\n{input_context_comment}\n[bold]Input current:[/bold] {args.input_current_text}"
        f"{output_context_comment}\n[bold]Output current:[/bold]\t{args.output_current_text}"
    )


def get_formatted_attribute_context_results(
    model: HuggingfaceModel,
    args: AttributeContextArgs,
    output: AttributeContextOutput,
    cti_threshold: float,
) -> str:
    """Format the results of the context attribution process."""

    def format_context_comment(
        model: HuggingfaceModel,
        has_other_context: bool,
        special_tokens_to_keep: list[str],
        context: str,
        context_scores: list[float],
        other_context_scores: Optional[list[float]] = None,
        is_target: bool = False,
        context_type: Literal["Input", "Output"] = "Input",
    ) -> str:
        context_tokens = get_filtered_tokens(
            context, model, special_tokens_to_keep, replace_special_characters=True, is_target=is_target
        )
        scores = context_scores
        if has_other_context:
            scores += other_context_scores
        context_ranked_tokens, threshold = filter_rank_tokens(
            tokens=context_tokens,
            scores=scores,
            std_threshold=args.attribution_std_threshold,
            topk=args.attribution_topk,
        )
        for idx, score, tok in context_ranked_tokens:
            context_tokens[idx] = f"[bold green]{tok}({score:.3f})[/bold green]"
        cci_threshold_comment = f"(CCI > {threshold:.3f})" if threshold is not None else ""
        # print(f"CCI Threshold: {threshold}")

        return f"\n[bold]{context_type} context {cci_threshold_comment}:[/bold]\t{''.join(context_tokens)}"

    out_string = ""
    output_current_tokens = get_filtered_tokens(
        output.output_current, model, args.special_tokens_to_keep, replace_special_characters=True, is_target=True
    )
    cti_theshold_comment = f"(CTI > {cti_threshold:.3f})" if cti_threshold is not None else ""
    for example_idx, cci_out in enumerate(output.cci_scores, start=1):
        curr_output_tokens = output_current_tokens.copy()
        cti_idx = cci_out.cti_idx
        cti_score = cci_out.cti_score
        cti_tok = curr_output_tokens[cti_idx]
        curr_output_tokens[cti_idx] = f"[bold dodger_blue1]{cti_tok}({cti_score:.3f})[/bold dodger_blue1]"
        output_current_comment = "".join(curr_output_tokens)
        input_context_comment, output_context_comment = "", ""
        if args.has_input_context:
            input_context_comment = format_context_comment(
                model,
                args.has_output_context,
                args.special_tokens_to_keep,
                output.input_context,
                cci_out.input_context_scores,
                cci_out.output_context_scores,
            )
        if args.has_output_context:
            output_context_comment = format_context_comment(
                model,
                args.has_input_context,
                args.special_tokens_to_keep,
                output.output_context,
                cci_out.output_context_scores,
                cci_out.input_context_scores,
                is_target=True,
                context_type="Output",
            )
        out_string += (
            f"#{example_idx}."
            f"\n[bold]Generated output {cti_theshold_comment}:[/bold]\t{output_current_comment}"
            f"{input_context_comment}{output_context_comment}\n"
        )
    return out_string


def visualize_attribute_context(
    output: AttributeContextOutput,
    model: Union[HuggingfaceModel, str, None] = None,
    cti_threshold: Optional[float] = None,
    return_html: bool = False,
) -> Optional[str]:
    if output.info is None:
        raise ValueError("Cannot visualize attribution results without args. Set add_output_info = True.")
    console = Console(record=True)
    viz = get_formatted_procedure_details(output.info)
    if model is None:
        model = output.info.model_name_or_path
    if isinstance(model, str):
        model = load_model(
            output.info.model_name_or_path,
            output.info.attribution_method,
            model_kwargs=deepcopy(output.info.model_kwargs),
            tokenizer_kwargs=deepcopy(output.info.tokenizer_kwargs),
        )
    elif not isinstance(model, HuggingfaceModel):
        raise TypeError(f"Unsupported model type {type(model)} for visualization.")
    if cti_threshold is None and len(output.cti_scores) > 1:
        cti_threshold = get_scores_threshold(output.cti_scores, output.info.context_sensitivity_std_threshold)
    viz += "\n\n" + get_formatted_attribute_context_results(model, output.info, output, cti_threshold)
    with console.capture() as _:
        console.print(viz, soft_wrap=False)
    html = console.export_html()
    if output.info.viz_path:
        with open(output.info.viz_path, "w", encoding="utf-8") as f:
            f.write(html)
    if output.info.show_viz:
        console.print(viz, soft_wrap=False)
    if return_html:
        return html
    return None


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import os

def visualize_image_context(image_path, cci_scores, cci_step_idx, target_word, save_path, args, img_shape, patch_shape, n_patches, save=False):
    """
    img_shape: to what shape the model reshapes the image.
    patch_shape: shape of the model patches.
    grid_size: how many patches are on the grid (n_hor, n_ver)
    """
    cci_scores_np = np.array(cci_scores)
    mean_ = cci_scores_np.mean()
    sd_ = cci_scores_np.std()
    threshold = mean_ + args.attribution_std_threshold * sd_
    cci_scores_threshold = [x for x in cci_scores if x > threshold] # Scores above the threshold: Save only those.
    
    highlight_list = np.where(cci_scores_np > threshold, 1, 0)

    # Load and resize the image
    image = Image.open(image_path).resize(img_shape) 
    width, height = img_shape
    
    # Calculate the size of each grid square
    n_patches_x, n_patches_y = n_patches
    patch_size_x, patch_size_y = patch_shape
    #grid_size_x = width // 16
    #grid_size_y = height // 16
    
    # Convert the image to a numpy array
    image_np = np.array(image)
    
    # Create a plot
    fig, ax = plt.subplots()
    ax.imshow(image_np)
    
    # Prepare to save bounding box coordinates of the patches corresponding to above threshold entries.
    # all_bbox_coordinates = []
    bbox_coordinates = []

    for i in range(n_patches_x):
        for j in range(n_patches_y):
            x = j * patch_size_x
            y = i * patch_size_y

            if highlight_list[i * patch_size_y + j] == 1:
                rect = patches.Rectangle(
                    (x, y),
                    patch_size_x,
                    patch_size_y,
                    linewidth=1,
                    edgecolor='r',
                    facecolor='r',
                    alpha=0.3
                )
                ax.add_patch(rect)
                
                # Save bounding box coordinates (top-left and bottom-right corners)
                bbox_coordinates.append((x, y, x + patch_size_x, y + patch_size_y))
            else:
                rect = patches.Rectangle(
                    (x, y),
                    patch_size_x,
                    patch_size_y,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none'
                )
                ax.add_patch(rect)
    
    # Save image with bboxes (not good for future as whenever I get two target words that are identical they will be overridden and become useless).
    save_path_image = f"{save_path}/images/target_{target_word}.png"
    plt.savefig(save_path_image)
    plt.close()
    if save == False:
        return
    # Also save the bbox of the identified squares (TODO: Make it model independent, for PaliGemma squares are read left down but others also have different crops!
    # DEMETRA
    # bboxes_file_path = '/u/sdavenia/VLM_Experiments/wildreceipts_evaluation/bboxes' 
    # LEONARDO
    bboxes_file_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/bboxes'
    os.makedirs(bboxes_file_path, exist_ok=True)
    bboxes_txt_file = os.path.join(bboxes_file_path, f"step{cci_step_idx}_{target_word}_bboxes.txt")
    
    with open(bboxes_txt_file, 'w') as f:
        for bbox in bbox_coordinates:
            f.write(f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}\n")

    # Also save above threshold cci_scores
    # DEMETRA
    # cci_file_path = '/u/sdavenia/VLM_Experiments/wildreceipts_evaluation/cci_scores' 
    # LEONARDO
    cci_file_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/cci_scores'
    os.makedirs(cci_file_path, exist_ok=True)
    cci_scores_txt_file = os.path.join(cci_file_path, f"step{cci_step_idx}_{target_word}_cci_scores.txt")
    
    with open(cci_scores_txt_file, 'w') as f:
        for cci_score in cci_scores_threshold:
            f.write(f"{cci_score}\n")
    #print(f"Bbox coordinates saved to: {txt_file}")
