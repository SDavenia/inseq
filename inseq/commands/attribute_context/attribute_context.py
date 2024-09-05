"""Implementation of the context attribution process described in `Quantifying the Plausibility of Context Reliance in
Neural Machine Translation <https://arxiv.org/abs/2310.01188>`_ for decoder-only and encoder-decoder models.

The process consists of two steps:
    - Context-sensitive Token Identification (CTI): detects which tokens in the generated output of interest are
        influenced by the presence of context.
    - Contextual Cues Imputation (CCI): attributes the generation of context-sensitive tokens identified in the first
        step to the input and output contexts.

Example usage:

```bash
inseq attribute-context \
    --model_name_or_path gpt2 \
    --input_context_text "George was sick yesterday." \
    --input_current_text "His colleagues asked him" \
    --attributed_fn contrast_prob_diff
```

PaliGemma-224
python -m inseq.commands.cli attribute-context --input_current_text "Describe this image" --attributed_fn contrast_prob_diff --model_name "google/paligemma-3b-mix-224" --generation_kwargs='{"max_new_tokens": 50}' --context_image_path /u/dssc/sdaven00/inseq/extra_samu/data/image_test.png
Leonardo path
python3 -m inseq.commands.cli attribute-context --input_current_text "Describe this image" --attributed_fn contrast_prob_diff --model_name "google/paligemma-3b-pt-224" --generation_kwargs='{"max_new_tokens": 50}' --context_image_path /leonardo/home/userexternal/sdavenia/inseq_dir/inseq/extra_samu/data/test_image.png


Leonardo path and use probability and not contrastive
python3 -m inseq.commands.cli attribute-context --input_current_text "Describe this image" --attributed_fn probability --model_name "google/paligemma-3b-mix-224" --generation_kwargs='{"max_new_tokens": 50}' --context_image_path /leonardo/home/userexternal/sdavenia/inseq_dir/inseq/extra_samu/data/test_image.png



PaliGemma-448
python -m inseq.commands.cli attribute-context --input_current_text "Describe this image" --attributed_fn contrast_prob_diff --model_name "google/paligemma-3b-mix-448" --generation_kwargs='{"max_new_tokens": 50}' --context_image_path /u/dssc/sdaven00/inseq/extra_samu/data/image_test.png
"""

import json
import logging
import warnings
from copy import deepcopy

import transformers

from ... import load_model
from ...attr.step_functions import is_contrastive_step_function
from ...models import HuggingfaceModel
from ..attribute import aggregate_attribution_scores
from ..base import BaseCLICommand
from .attribute_context_args import AttributeContextArgs
from .attribute_context_helpers import (
    AttributeContextOutput,
    CCIOutput,
    concat_with_sep,
    filter_rank_tokens,
    format_template,
    get_contextless_output,
    get_filtered_tokens,
    get_source_target_cci_scores,
    prepare_outputs,
)
from .attribute_context_viz_helpers import visualize_attribute_context, visualize_image_context

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def attribute_context(args: AttributeContextArgs) -> AttributeContextOutput:
    """Attribute the generation of context-sensitive tokens in ``output_current_text`` to input/output contexts."""
    import torch
    import pandas as pd
    # TODO METTI IL DATASET GIUSTO
    base_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation'
    # paligemma-3b-mix-224
    # df = pd.read_pickle(f"{base_path}/wildreceipts_correct_paligemma-3b-mix-224.pkl").reset_index()
    # paligemma-3b-pt-224 finetuned
    df = pd.read_pickle(f"{base_path}/wildreceipts_correct_paligemma-3b-pt-224_wildreceipts_the_price_is.pkl").reset_index()
    print(f"Loading model... to device {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    from PIL import Image
    import tempfile
    from copy import deepcopy as dp
    import os
    import re
    # input_current_text
    # context_image_path
    model: HuggingfaceModel = load_model(
        args.model_name_or_path,
        args.attribution_method,
        model_kwargs=deepcopy(args.model_kwargs),
        tokenizer_kwargs=deepcopy(args.tokenizer_kwargs),
    )
    # FOR LOADING PRETRAINED MODEL INSTEAD
    print(f"Succesfully loaded model")
    from transformers import PaliGemmaForConditionalGeneration
    import torch
    # TODO CHECK
    # DEMETRA
    # pretrained_model_path = '/u/sdavenia/VLM_Experiments/wildreceipts_evaluation/verbose_model/ft_checkpoints/no_visionno_projectorpaligemma-3b-pt-224_wildreceipts_the_price_is.hf/checkpoint-150'
    # LEONARDO
    # pretrained_model_path = '/leonardo_scratch/fast/IscrC_XAI-MRAG/multimodal_pecore/ft_checkpoints/no_visionno_projectorpaligemma-3b-pt-224_wildreceipts_the_price_is.hf/checkpoint-150'
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model.model = PaliGemmaForConditionalGeneration.from_pretrained(pretrained_model_path).to(device)

    def save_image_temp(image):
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        image.save(temp_file, format="JPEG")
        temp_file.close()
        return temp_file.name
    
    # Files where temp are saved
    cti_scores_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/cti_scores'
    cci_scores_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/cci_scores'
    bboxes_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/bboxes'

    # File where to save results
    model_name = re.search(r'[^/]+$', args.model_name_or_path).group(0)
    contrastive_type_str = 'black' if args.attributed_fn == 'contrast_prob_diff' or args.attributed_fn == 'kl_divergence' else 'None'
    ctistd_str = str(args.context_sensitivity_std_threshold) if args.context_sensitivity_std_threshold > -10 else 'all'
    ccistd_str = str(args.attribution_std_threshold) if args.attribution_std_threshold > -10 else 'all'
    # TODO FIX CHANGE
    # If there is NOT ft dataset
    base_save_path = f"/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/wildreceipts_pecore_results_{model_name}_{contrastive_type_str}_{args.attributed_fn}_ctistd_{ctistd_str}_ccistd_{ccistd_str}"
    # If there is a ft dataset
    # ft_df = 'wildreceipts_the_price_is'
    # base_save_path = f"/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/wildreceipts_pecore_results_{model_name}_{ft_df}_{contrastive_type_str}_{args.attributed_fn}_ctistd_{ctistd_str}_ccistd_{ccistd_str}"
    save_steps = 300
    all_cti_tokens_list = []
    all_bboxes_list = []
    all_cci_scores_list = []
    all_cti_scores_list = []

    # For temporary saves
    temp_save_counter = 0
    temp_cti_tokens_list = []
    temp_bboxes_list = []
    temp_cci_scores_list = []
    temp_cti_scores_list = []

    # Restart from where you finished before and finish running -> Just have to save last one
    for idx, row in df.iterrows():
        args_row = dp(args)
        # CHECK TODO MODIFY FT
        # args_row.input_current_text = f"long answer: What is the price of {row['item'].strip()}"
        args_row.input_current_text = f"What is the price of {row['item'].strip()}?"
        # args_row.input_current_text = f"Describe this image."
        temp_img_path = save_image_temp(row['image'])       
        args_row.context_image_path=temp_img_path
        #args_row.context_image_path = '/leonardo/home/userexternal/sdavenia/inseq_dir/inseq/extra_samu/data/dog.jpg'
        
        print(args_row.attributed_fn)
        attribute_context_with_model(args_row, model)

        # Lists to store words, bounding boxes and cci_scores
        cti_tokens_list = []
        bboxes_list = []
        cci_scores_list = []
            
        # Extract words names and bboxes.
        for file_name in os.listdir(bboxes_path):
            if file_name.endswith('_bboxes.txt'):
                # Extract the word from the filename
                #word = file_name.split('_bboxes.txt')[0]
                word = re.search(r"step\d+_(.*?)_bboxes.txt", file_name).group(1)
                cti_tokens_list.append(word)

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
        # Extract cti_scores from the saved file and remove the file.
        for file_name in os.listdir(cti_scores_path):
            cti_scores_file_path = os.path.join(cti_scores_path, file_name)
            current_cti_scores = []
            with open(cti_scores_file_path, 'r') as file:
                for line in file:
                    score = line.strip()
                    current_cti_scores.append(score)
            cti_scores_list = current_cti_scores
            os.remove(cti_scores_file_path)
        
        # lists with all
        all_cti_tokens_list.append(cti_tokens_list)
        all_cti_scores_list.append(cti_scores_list)
        all_bboxes_list.append(bboxes_list)
        all_cci_scores_list.append(cci_scores_list)
        # lists with temporary saves
        temp_cti_tokens_list.append(cti_tokens_list)
        temp_cti_scores_list.append(cti_scores_list)
        temp_bboxes_list.append(bboxes_list)
        temp_cci_scores_list.append(cci_scores_list)
        if (idx + 1) % save_steps == 0:
            df_temp = pd.DataFrame({
                'cti_tokens': temp_cti_tokens_list,
                'cti_scores': temp_cti_scores_list,
                'bboxes': temp_bboxes_list,
                'cci_scores': temp_cci_scores_list
            })
            # Reset
            temp_cti_tokens_list = []
            temp_bboxes_list = []
            temp_cci_scores_list = []
            temp_cti_scores_list = []
            # Save 
            temp_csv_path = f"{base_save_path}_temp_df_{temp_save_counter}.csv"
            print(temp_csv_path)
            df_temp.to_csv(temp_csv_path, index=False)
            temp_save_counter += 1
    df['cti_tokens'] = all_cti_tokens_list
    df['cti_scores'] = all_cti_scores_list
    df['bboxes'] = all_bboxes_list
    df['cci_scores'] = all_cci_scores_list
    
    final_df_path = f"{base_save_path}.pkl"
    df.to_pickle(final_df_path)
    print(f"Saved to: {final_df_path}")
    """
    return attribute_context_with_model(args, model)
    """


def attribute_context_with_model(args: AttributeContextArgs, model: HuggingfaceModel) -> AttributeContextOutput:
    # Handle language tag for multilingual models - no need to specify it in generation kwargs
    has_lang_tag = "tgt_lang" in args.tokenizer_kwargs
    if has_lang_tag and "forced_bos_token_id" not in args.generation_kwargs:
        tgt_lang = args.tokenizer_kwargs["tgt_lang"]
        args.generation_kwargs["forced_bos_token_id"] = model.tokenizer.lang_code_to_id[tgt_lang]

    # Prepare input/outputs (generate if necessary)
    # Prepare input image: if model is vlm and image is not passed raised an error.
    if model.is_vlm:
        if not args.context_image and not args.context_image_path:
            raise ValueError("Using a VLM requires an image to be passed as input.")
        if not args.context_image:
            args.context_image = transformers.image_utils.load_image(args.context_image_path)
    #print(f"Preparing input/output (generate if necessary)")
    #print(f"    Calling format_template on the input")
    # input_full_text = input_context_text + input_current_text
    input_full_text = format_template(args.input_template, args.input_current_text, args.input_context_text)
    #print(f"    Calling prepare outputs")
    args.output_context_text, args.output_current_text = prepare_outputs(
        model=model,
        input_context_text=args.input_context_text,
        input_full_text=input_full_text,
        output_context_text=args.output_context_text,
        output_current_text=args.output_current_text,
        output_template=args.output_template,
        handle_output_context_strategy=args.handle_output_context_strategy,
        input_context_image = args.context_image,                               # Added context image needed for contextual generation with vlms
        generation_kwargs=deepcopy(args.generation_kwargs),
        special_tokens_to_keep=args.special_tokens_to_keep,
        decoder_input_output_separator=args.decoder_input_output_separator,
    )
    #print(f"    Calling format template on the output")
    output_full_text = format_template(args.output_template, args.output_current_text, args.output_context_text)

    # Remove unnecessary special tokens -> We just cleanup no tokenization occurring yet.
    input_context_tokens = None
    if args.input_context_text is not None:
        input_context_tokens = get_filtered_tokens(args.input_context_text, model, args.special_tokens_to_keep)

    if not model.is_encoder_decoder:
        output_full_text = concat_with_sep(input_full_text, output_full_text, args.decoder_input_output_separator)
        # print(f"now output_full_text is : {output_full_text}")
    output_current_tokens = get_filtered_tokens(
        args.output_current_text, model, args.special_tokens_to_keep, is_target=True
    )
    output_context_tokens = None
    if args.output_context_text is not None:
        output_context_tokens = get_filtered_tokens(
            args.output_context_text, model, args.special_tokens_to_keep, is_target=True
        )
    # Manually add \n 
    # TODO: Make it nicer and not manual
    # TODO: This is probably only for PaliGemma
    if model.is_vlm:
        input_full_text = input_full_text.strip() + '\n'
        output_full_text = output_full_text[:len(input_full_text)-1].strip() + '\n' + output_full_text[len(input_full_text)-1:].strip()
    #print(f"input full text: {repr(input_full_text)}")
    #print(f"output full text: {repr(input_full_text)}")

    input_full_tokens = get_filtered_tokens(input_full_text, model, args.special_tokens_to_keep)
    output_full_tokens = get_filtered_tokens(output_full_text, model, args.special_tokens_to_keep, is_target=True)
    output_current_text_offset = len(output_full_tokens) - len(output_current_tokens)
    #print(f"Output current_tokens: {output_current_tokens}")
    formatted_input_current_text = args.contextless_input_current_text.format(current=args.input_current_text)
    formatted_output_current_text = args.contextless_output_current_text.format(current=args.output_current_text)
    if not model.is_encoder_decoder:
        formatted_input_current_text = concat_with_sep(
            formatted_input_current_text, "", args.decoder_input_output_separator
        )
        formatted_output_current_text = formatted_input_current_text + formatted_output_current_text

    # Manually add \n between input and model generation (PaliGemma processor does it, but it seems to get lost in the processing above)
    # TODO: Make it nicer and not manual.
    if model.is_vlm:
        formatted_output_current_text = formatted_output_current_text[:len(formatted_input_current_text)].strip() + '\n' + formatted_output_current_text[len(formatted_input_current_text):].strip()
        formatted_input_current_text = formatted_input_current_text.strip() + '\n'
    #    output_full_text = formatted_output_current_text
    # print(f"formatted_input_current_text: {repr(formatted_input_current_text)}")
    # print(f"formatted_output_current_text: {repr(formatted_output_current_text)}")
    # print(f"output full text: {repr(output_full_text)}")

    # Part 1: Context-sensitive Token Identification (CTI)
    #print(f"\n\nCTI")
    #print(f"model.attribute is called with the following parameters:")
    #print(f"\tInput texts: {repr(formatted_input_current_text)}")
    #print(f"\tGenerated texts: {repr(formatted_output_current_text)}")
    #print(f"\tContrast targets: {repr(formatted_output_current_text)}")
    # print(f"Model is: {model}")

    cti_out = model.attribute(
        input_texts=formatted_input_current_text.rstrip(" "),
        generated_texts=formatted_output_current_text,
        # context_image=args.context_image, # Added it again as an arg for attribute -> check later.
        attribute_target=model.is_encoder_decoder,
        step_scores=[args.context_sensitivity_metric],
        contrast_sources=input_full_text if model.is_encoder_decoder else None,
        contrast_targets=output_full_text,
        show_progress=False,
        method="dummy",
        context_image = args.context_image, # Now it is a kwarg specific for VLM models (in the definition)
    )[0]
    if args.show_intermediate_outputs:
        # print(f"SHOWING THEM")
        cti_out.show(do_aggregation=False)
    start_pos = 1 if has_lang_tag else 0
    contextless_output_prefix = args.contextless_output_current_text.split("{current}")[0]
    contextless_output_prefix_tokens = get_filtered_tokens(
        contextless_output_prefix, model, args.special_tokens_to_keep, is_target=True
    )
    start_pos += len(contextless_output_prefix_tokens)
    cti_scores = cti_out.step_scores[args.context_sensitivity_metric][start_pos:].tolist()
    cti_tokens = [t.token for t in cti_out.target][start_pos + cti_out.attr_pos_start :]
    if model.is_encoder_decoder:
        cti_scores = cti_scores[:-1]
        cti_tokens = cti_tokens[:-1]
    #print(f"cti tokens: {cti_tokens}")
    #print(f"cti scores: {cti_scores}")
    # For paligemma last token is \n generation -> Remove it from CTI for my experiments for now.
    if args.model_name_or_path == 'google/paligemma-3b-mix-224' or args.model_name_or_path == 'google/paligemma-3b-mix-448' or args.model_name_or_path == 'google/paligemma-3b-pt-224' or args.model_name_or_path == 'google/paligemma-3b-pt-448':
        cti_tokens = cti_tokens[:-1]
        cti_scores = cti_scores[:-1]
    cti_ranked_tokens, cti_threshold = filter_rank_tokens(
        tokens=cti_tokens,
        scores=cti_scores,
        std_threshold=args.context_sensitivity_std_threshold,
        topk=args.context_sensitivity_topk,
    )   

    # YESSAVE
    import os
    # No need to change since these are only stored temporally to be read back by the model.
    # DEMETRA
    # cti_scores_path = '/u/sdavenia/VLM_Experiments/wildreceipts_evaluation/cti_scores'
    # LEONARDO
    cti_scores_path = '/leonardo/home/userexternal/sdavenia/VLM_experiments_dir/VLM_Experiments/wildreceipts_evaluation/cti_scores'
    os.makedirs(cti_scores_path, exist_ok=True)
    cti_scores_file = os.path.join(cti_scores_path, f"cti_scores.txt")
    with open(cti_scores_file, 'w') as f:
        for item in cti_ranked_tokens:
            score = item[1]
            f.write(f"{score}\n")

    output = AttributeContextOutput(
        input_context=args.input_context_text,
        input_context_tokens=input_context_tokens,
        output_context=args.output_context_text,
        output_context_tokens=output_context_tokens,
        output_current=args.output_current_text,
        output_current_tokens=output_current_tokens,
        cti_scores=cti_scores,
        info=args,
    )
    # Part 2: Contextual Cues Imputation (CCI)
    print(f"CTI ranked tokens: {cti_ranked_tokens}")
    print(f"Entering CCI...")

    # Iterate over all context sensitive generated tokens.
    for cci_step_idx, (cti_idx, cti_score, cti_tok) in enumerate(cti_ranked_tokens):        
        # print(f"Processing token {cti_idx} with score {cti_score} and token {cti_tok}")
        contextual_input = model.convert_tokens_to_string(input_full_tokens, skip_special_tokens=False).lstrip(" ")
        # print(f"HERE Output_full_tokens: {output_full_tokens}") # 'George','was', 'sick', 'yesterday' ... fino alla fine
        contextual_output = model.convert_tokens_to_string(
            output_full_tokens[: output_current_text_offset + cti_idx + 1], skip_special_tokens=False
        ).lstrip(" ")
        #print(f"Contextual input: {repr(contextual_input)}")      # Contains (string) context + input. "George was sick yesterday. His colleagues asked him how "
                                                                   # For VLM Describe this image\n
        #print(f"Contextual output: {repr(contextual_input)}")     # Contains context + input + generation up to target token (included) "George was sick yesterday. His colleagues asked him how he was **doing**"
                                                                   # For VLM Describe this image\ncon
        if not contextual_output: # If there is no output generated (not really sure how this could occur)
            output_ctx_tokens = [output_full_tokens[output_current_text_offset + cti_idx]]
            if model.is_encoder_decoder:
                output_ctx_tokens.append(model.pad_token)
            contextual_output = model.convert_tokens_to_string(output_ctx_tokens, skip_special_tokens=True)
        else:
            output_ctx_tokens = model.convert_string_to_tokens(
                contextual_output, skip_special_tokens=False, as_targets=model.is_encoder_decoder
            )
        #print(f"output_ctx_tokens:{repr(output_ctx_tokens)}")   # Contains individual tokens of contextual output up to current attribution step ['George', 'was', 'sick', ... 'was', 'doing']  
                                                                # For VLM ['Describe', '▁this', '▁image', '\n', 'con']
        """
        CONTROLLA POI COME SI COMPORTA LA GENERAZIONE QUA SE HAI UN TOKEN IN MEZZO
        In input vanno i modificati del tutto?
        """
        cci_kwargs = {}
        contextless_output = None
        #print(f"args.attributed_fn: {args.attributed_fn}")  # Contains attributed_fn: in out case contrast_prob_diff
        print(is_contrastive_step_function(args.attributed_fn))
        if args.attributed_fn is not None and is_contrastive_step_function(args.attributed_fn):
            #print(f"Using a contrastive step function:")
            if not model.is_encoder_decoder:
                # In our case remains the same since contextless_output_prefix is empty since we are not using nested prefixes
                
                # args.decoder_input_output_separator: For PaliGemma VLM it should be the \n (maybe it would make sense to add it here but for now ignore)
                #  since we have added it manually above.
                if not model.is_vlm:
                    formatted_input_current_text = concat_with_sep(
                        formatted_input_current_text, contextless_output_prefix, args.decoder_input_output_separator # input
                    )
            # print(f"Args.contextless_output_next_tokens:\n{args.contextless_output_next_tokens}") # Empty: Not sure if that is something that to be there the user has to specify manually.
            #print(f"Calling get_contextless_output with:")
            #print(f"    formatted_input_current_text: {repr(formatted_input_current_text)}")
            #print(f"    output_current_tokens: {repr(output_current_tokens)}") # Contains full output
            contextless_output = get_contextless_output(    # Ends up calling model generate with only input (for VLM ENSURE black image is passed here in some way, since probs you are passing the image itself.)
                model,
                formatted_input_current_text,    # His colleagues asked him how (input only)
                                                 # Describe this image\n
                output_current_tokens,           # 
                cti_idx,
                cti_ranked_tokens,
                args.contextless_output_next_tokens,
                args.prompt_user_for_contextless_output_next_tokens,
                cci_step_idx,
                args.decoder_input_output_separator,
                args.special_tokens_to_keep,
                deepcopy(args.generation_kwargs),
            )
            #print(f"Formatted input current text: {repr(formatted_input_current_text)}")
            if "\n\n" in contextless_output:
                contextless_output = contextless_output.replace('\n\n', '\n')
            #print(f"Contextless output: {repr(contextless_output)}") # String containing generation without context.
                                                               # For unimodal example it appears to be the same as contextual case.
                                                               # For VLM model it is Describe this image\nun 
            cci_kwargs["contrast_sources"] = formatted_input_current_text if model.is_encoder_decoder else None
            cci_kwargs["contrast_targets"] = contextless_output
            output_ctxless_tokens = model.convert_string_to_tokens(
                contextless_output, skip_special_tokens=False, as_targets=model.is_encoder_decoder
            )
            tok_pos = -2 if model.is_encoder_decoder else -1
            #print(f"output_ctx_tokens: {repr(output_ctx_tokens[tok_pos])}")           # Next token when generating with context (for VLM con)
            #print(f"output_ctxless_tokens: {repr(output_ctxless_tokens[tok_pos])}")   # Next token when generating without contextless (for VLM un)

            # If we are using kl divergence for attributed_fn or if the token is the same in the context and contextless output.
            if args.attributed_fn == "kl_divergence" or output_ctx_tokens[tok_pos] == output_ctxless_tokens[tok_pos]:
                #print(f"Setting contrast_force_inputs: True")
                cci_kwargs["contrast_force_inputs"] = True
        bos_offset = int(model.is_encoder_decoder or output_ctx_tokens[0] == model.bos_token)
        #print(f"output_current_text_offset: {output_current_text_offset}") # [0: describe, 1: this, 2: image, 3:\n, 4: un]
        #print(f"cti_idx: {cti_idx}") 
        #print(f"bos_offset: {bos_offset}")
        pos_start = output_current_text_offset + cti_idx + bos_offset + int(has_lang_tag)
        # TODO: Fix pos_start in a nicer way than hard-coding it like here
        if model.is_vlm:
            if args.model_name_or_path == 'google/paligemma-3b-mix-224' or args.model_name_or_path == 'google/paligemma-3b-pt-224':
                bos_offset = 1
                img_tokens = 256
                pos_start = pos_start + bos_offset + img_tokens
            elif args.model_name_or_path == 'google/paligemma-3b-mix-448' or args.model_name_or_path == 'google/paligemma-3b-pt-448':
                bos_offset = 1
                img_tokens = 1024
                pos_start = pos_start + bos_offset + img_tokens
            else:
                raise ValueError("At the moment model specific implementations work only for paligemma models.")

        # Add context image to cci_kwargs        
        # Need to find a way to pass image but not in the same way as before as we need it for batch and not for contrast_batch.
        if model.is_vlm:
            cci_kwargs['cci_context_image'] = args.context_image
            
        #print(f"Calling model attribute with:")
        #print(f"    Contextual input: {repr(contextual_input)}")    # context + input
        #print(f"    Contextual output: {repr(contextual_output)}")  # context + input + output (generated with context)
        #print(f"    Position start: {pos_start}")                   # 12: position of the token currently being investigated
        #print(f"    Attributed function: {args.attributed_fn}")     # contrast_prob_diff
        #print(f"    Attribution method: {args.attribution_method}") # saliency
        #print(f"    CCI Kwargs: {cci_kwargs}")                      # contrast_sources: only for encoder decoder I believe.
                                                                    # contrast_targets: input + generation up to CTI token (obtained without context).
                                                                    # contrast_force_inputs: True depending on how it was set above!
        #print(f"    Args.attribution_kwargs: {args.attribution_kwargs}") # {}
        cci_attrib_out = model.attribute(
            contextual_input,
            contextual_output,
            attribute_target=model.is_encoder_decoder and args.has_output_context,
            show_progress=False,
            attr_pos_start=pos_start,
            attributed_fn=args.attributed_fn,
            method=args.attribution_method,
            cci=1,
            **cci_kwargs,
            **args.attribution_kwargs,
        )
        # print(f"cci_attrib_out:\n{cci_attrib_out}")
        # Below we extract the gradients that we're interested in. I believe it simply aggregates 
        #print(f"selectors: {args.attribution_selectors}") # None
        #print(f"aggregators: {args.attribution_aggregators}")  # None
        #print(f"normalize_attributions: {args.normalize_attributions}") # False
        cci_attrib_out = aggregate_attribution_scores(
            out=cci_attrib_out,
            selectors=args.attribution_selectors,
            aggregators=args.attribution_aggregators,
            normalize_attributions=args.normalize_attributions,
        )[0]
        #print(f"cci_attrib_out:\n{cci_attrib_out}")
        #print(f"cci_target_attributions:\n{cci_attrib_out.target_attributions}")
        #print(f"Len cci target attributions: {len(cci_attrib_out.target_attributions)}")
        if args.show_intermediate_outputs:
            cci_attrib_out.show(do_aggregation=False)
        source_scores, target_scores = get_source_target_cci_scores(
            model,
            cci_attrib_out,
            args.input_template,
            args.input_current_text,
            input_context_tokens,
            input_full_tokens,
            args.output_template,
            output_context_tokens,
            args.has_input_context,
            args.has_output_context,
            has_lang_tag,
            args.decoder_input_output_separator,
            args.special_tokens_to_keep,
        )
        #print(f"source scores: {len(source_scores)}\n{source_scores}") # Should contain scors for the target tokens
        #print(f"target scores: {target_scores}") # None for decoder only models
        cci_out = CCIOutput(
            cti_idx=cti_idx,
            cti_token=cti_tok,
            cti_score=cti_score,
            contextual_output=contextual_output,
            contextless_output=contextless_output,
            input_context_scores=source_scores,
            output_context_scores=target_scores,
        )
        # TODO: FA SCHIFO scritto cosi
        if model.is_vlm:
            if args.model_name_or_path == 'google/paligemma-3b-mix-224' or args.model_name_or_path == 'google/paligemma-3b-pt-224':
                cci_out.contextual_output =  '<img>' * 256 + cci_out.contextual_output
                if cci_out.contextless_output is not None:
                    cci_out.contextless_output =  '<img>' * 256 + cci_out.contextless_output
            elif args.model_name_or_path == 'google/paligemma-3b-mix-448' or args.model_name_or_path == 'google/paligemma-3b-pt-448':
                cci_out.contextual_output =  '<img>' * 1024 + cci_out.contextual_output
                if cci_out.contextless_output is not None:
                    cci_out.contextless_output =  '<img>' * 1024 + cci_out.contextless_output
            else:
                raise ValueError("At the moment model specific implementations work only for paligemma models.")
        output.cci_scores.append(cci_out)
        #print(f"cci_out : {cci_out}")
        #print(f"cci_out.input_context_scores: {cci_out.input_context_scores}")
        # Save the image for VLM visualization
        #print(f"Target is: {cci_out.cti_token}")

        # Added save=False to avoid getting lost.
        if model.is_vlm:
            if args.model_name_or_path == 'google/paligemma-3b-mix-224' or args.model_name_or_path == 'google/paligemma-3b-pt-224':
                img_shape = (224, 224)
                patch_shape = (14, 14)
                img_tokens = (img_shape[0] * img_shape[1]) / (patch_shape[0] * patch_shape[1])
                n_patches_x = img_shape[0] / patch_shape[0] # Number of patches on the horizontal side of the image
                n_patches_y = img_shape[1] / patch_shape[1] # Number of patches on the vertical side of the image
                if n_patches_x != int(n_patches_x) or n_patches_y != int(n_patches_y):
                    raise ValueError("n_patches_x and n_patches_y assumed to be whole numbers (e.g., 32.0).")
                n_patches = (int(n_patches_x), int(n_patches_y))
            elif args.model_name_or_path == 'google/paligemma-3b-mix-448' or args.model_name_or_path == 'google/paligemma-3b-pt-448':
                img_shape = (448, 448)
                patch_shape = (14, 14)
                img_tokens = (img_shape[0] * img_shape[1]) / (patch_shape[0] * patch_shape[1])
                n_patches_x = img_shape[0] / patch_shape[0]
                n_patches_y = img_shape[1] / patch_shape[1]
                if n_patches_x != int(n_patches_x) or n_patches_y != int(n_patches_y):
                    raise ValueError("n_patches_x and n_patches_y assumed to be whole numbers (e.g., 32.0).")
                n_patches = (int(n_patches_x), int(n_patches_y))
            else:
                raise ValueError("At the moment model specific implementations work only for paligemma models.")
        visualize_image_context(image_path = args.context_image_path, 
                                cci_scores = cci_out.input_context_scores, 
                                cci_step_idx=cci_step_idx,
                                target_word = cci_out.cti_token,
                                # DEMETRA
                                # save_path = '/u/sdavenia/inseq/extra_samu/wildreceipts_results',
                                # LEONARDO
                                save_path = '/leonardo/home/userexternal/sdavenia/inseq_dir/inseq/extra_samu/wildreceipts_results',
                                args=args,
                                img_shape = img_shape, 
                                patch_shape = patch_shape, 
                                n_patches = n_patches,
                                save=True) # YESSAVE: If this is False then only the img with the bboxes highlighted is saved
    
    if args.context_image is not None:
        # Stop here for VLM as no point in showing from terminal.
        return output
    if args.show_viz or args.viz_path and not model.is_vlm:
        visualize_attribute_context(output, model, cti_threshold)
    if not args.add_output_info:
        output.info = None
    if args.save_path:
        with open(args.save_path, "w") as f:
            json.dump(output.to_dict(), f, indent=4)
    return output


class AttributeContextCommand(BaseCLICommand):
    _name = "attribute-context"
    _help = "Detect context-sensitive tokens in a generated text and attribute their predictions to available context."
    _dataclasses = AttributeContextArgs

    def run(args: AttributeContextArgs):
        attribute_context(args)
