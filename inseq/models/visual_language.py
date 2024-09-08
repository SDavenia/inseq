import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

import PIL
import torch
import torchvision.transforms as transforms


from ..attr.feat import join_token_ids
from ..attr.step_functions import StepFunctionVLMArgs
from ..data import (
    BatchEmbedding,
    BatchEncoding,
    DecoderOnlyBatch,
    FeatureAttributionInput,
    FeatureAttributionStepOutput,
    get_batch_from_inputs,
)
from ..utils import get_aligned_idx
from ..utils.typing import (
    AttributionForwardInputs,
    EmbeddingsTensor,
    ExpandedTargetIdsTensor,
    IdsTensor,
    LogitsTensor,
    OneOrMoreTokenSequences,
    SingleScorePerStepTensor,
    TargetIdsTensor,
    TextSequences,
)
from .attribution_model import AttributionModel, ForwardMethod, InputFormatter, ModelOutput

CustomForwardOutput = TypeVar("CustomForwardOutput")

logger = logging.getLogger(__name__)

class VLMInputFormatter(InputFormatter):
    @staticmethod
    def prepare_inputs_for_attribution( 
        attribution_model: "VLMAttributionModel",
        inputs: FeatureAttributionInput, # FOR VLM INPUTS IS A TUPLE (str, image)
        include_eos_baseline: bool = False,
        skip_special_tokens: bool = False,
    ) -> DecoderOnlyBatch:
        """ 
        Prepares the input for attribution, i.e. it prepares a DecoderOnlyBatch object from the input.
        If input is only text, we add a black pixels image to it.
        This contains:
            - encoding: a BatchEncodingObject (input_ids, attention_mask, input_tokens, baseline_ids, pixel_values)
            - embedding: a BatchEmbeddingObject (input_embeds, baseline_embeds, black_image_embeds).
        """
        #print(f"Calling prepare_inputs_for_attributions (VLM Model)")
        #print(f"Before adding black image Inputs are: {inputs}")

        # Inputs can be:
        # - list containing a string -> We're doing CTI and add black image!
        # - (list containing a string, image) -> We're doing CCI!

        # Updated: Inputs can be:
        # - a tuple containing (string, image) where image can either be the full image or the contrastive image.

        # Need to add black image
        if not isinstance(inputs, tuple):
            raise ValueError("SHOULD NOT ENTER HERE NOW WE ARE USING CONTEXTLESS IMAGE.")
            black_image = PIL.Image.new("RGB", (100, 100), (0, 0, 0)) # Generate black image and pass it.
            inputs = (inputs, black_image)
        # print(f"inputs inside: {repr(inputs[0][0])}") Here additional \n is not present
        batch = get_batch_from_inputs(
            attribution_model,
            inputs=inputs, # To be called here inputs should be (textual_input, image) where textual input is a list of strings and image can either be the context or the contextless image
            include_eos_baseline=include_eos_baseline,
            as_targets=False,
            skip_special_tokens=skip_special_tokens,
        )
        return DecoderOnlyBatch.from_batch(batch)

    # LEFT THE SAME AS DECODER ONLY ARGS: AS WE WILL HAVE THEM WORKING IN THE SAME WAY
    @staticmethod
    def get_text_sequences(attribution_model: "VLMAttributionModel", batch: DecoderOnlyBatch) -> TextSequences:
        return TextSequences(
            sources=None,
            targets=attribution_model.decode(batch.target_ids),
        )

    # LEFT THE SAME AS DECODER ONLY ARGS: AS WE WILL HAVE THEM WORKING IN THE SAME WAY
    @staticmethod
    def get_step_function_reserved_args() -> list[str]:
        return [f.name for f in StepFunctionVLMArgs.__dataclass_fields__.values()]
    
    # LEFT THE SAME AS DECODER ONLY ARGS: AS WE WILL HAVE THEM WORKING IN THE SAME WAY
    @staticmethod
    def format_attribution_args(
        batch: DecoderOnlyBatch,
        target_ids: TargetIdsTensor,
        attributed_fn: Callable[..., SingleScorePerStepTensor],
        attribute_target: bool = False,  # Needed for compatibility with EncoderDecoderAttributionModel
        attributed_fn_args: dict[str, Any] = {},
        attribute_batch_ids: bool = False,
        forward_batch_embeds: bool = True,
        use_baselines: bool = False,
    ) -> tuple[dict[str, Any], tuple[Union[IdsTensor, EmbeddingsTensor, None], ...]]:
        """
        This function is called for every step of the attribution process (i.e. for each generated token after the input.) to format the args.
        It returns a dictionary containing:
            - inputs: i.e. the embeddings of the ids of the input + generated tokens so far
            - additional_forward_args: a tuple containing:
                - input_ids of the input + generated tokens so far
                - target_ids: the target token id (i.e. the token that was force generated in this step)
        """
        # print(f"Calling format_attribution_args (VLM only)")
        if attribute_batch_ids:
            inputs = (batch.input_ids,)
        else:
            #print(f"format attribution using the embeddings.")
            inputs = (batch.input_embeds)
            # print(f"inputs: {batch.input_embeds[0, 5, :10]}")
            # baselines = (batch.baseline_ids,)
        #else:
        #    print(f"Setting black embeds.")
        #    inputs = (batch.black_embeds,)      # MODIFIED: WE TAKE THE 
        #    #baselines = (batch.baseline_embeds,)
        attribute_fn_args = {
            "inputs": inputs,
            "additional_forward_args": (
                # Ids are always explicitly passed as extra arguments to enable
                # usage in custom attribution functions.
                batch.input_ids,
                # Making targets 2D enables _expand_additional_forward_args
                # in Captum to preserve the expected batch dimension for methods
                # such as intergrated gradients.
                target_ids.unsqueeze(-1),
                attributed_fn,
                batch.attention_mask,
                # Defines how to treat source and target tensors
                # Maps on the use_embeddings argument of forward
                forward_batch_embeds,
                list(attributed_fn_args.keys()),
            )
            + tuple(attributed_fn_args.values()),
        }
        #if use_baselines:
        #    attribute_fn_args["baselines"] = baselines
        #print(f"Returning: attribute_fn_args")
        #print(f"    Inputs have shape: {attribute_fn_args['inputs'][0].shape}")
        #print(f"    Additional forward args are:\n{attribute_fn_args['additional_forward_args']}")
        return attribute_fn_args
    
    # LEFT THE SAME AS DECODER ONLY ARGS: AS WE WILL HAVE THEM WORKING IN THE SAME WAY
    @staticmethod
    def format_step_function_args(
        attribution_model: "VLMAttributionModel",
        forward_output: ModelOutput,
        target_ids: ExpandedTargetIdsTensor,
        batch: DecoderOnlyBatch,
        is_attributed_fn: bool = False,
    ) -> StepFunctionVLMArgs:
        return StepFunctionVLMArgs(
            attribution_model=attribution_model,
            forward_output=forward_output,
            target_ids=target_ids,
            is_attributed_fn=is_attributed_fn,
            decoder_input_ids=batch.target_ids,
            decoder_attention_mask=batch.target_mask,
            decoder_input_embeds=batch.target_embeds,
            context_image=None,
            contextless_image=None
            # context_image=transforms.ToPILImage(mode='RGB')(batch.encoding.pixel_values[0])
        )

    # NEEDS TO BE MODIFIED TO ALSO HAVE IMAGE
    @staticmethod
    def convert_args_to_batch(
        args: StepFunctionVLMArgs = None,
        decoder_input_ids: Optional[IdsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        **kwargs,
    ) -> DecoderOnlyBatch:
        print(f"Calling convert_args_to_batch (VLM)")
        #print(f"Args is:\n{args}")

        #print(f"Decoder input ids: {decoder_input_ids}")
        #print(f"Decoder attention_mask: {decoder_attention_mask}")
        #print(f"Decoder input_embeds: {decoder_input_embeds}")
        if args is not None:
            decoder_input_ids = args.decoder_input_ids
            decoder_attention_mask = args.decoder_attention_mask
            decoder_input_embeds = args.decoder_input_embeds
        encoding = BatchEncoding(decoder_input_ids, decoder_attention_mask)
        embedding = BatchEmbedding(decoder_input_embeds)
        return DecoderOnlyBatch(encoding, embedding)
    
    # SAME AS DECODER ONLY
    @staticmethod
    def enrich_step_output(
        attribution_model: "VLMAttributionModel",
        step_output: FeatureAttributionStepOutput,
        batch: DecoderOnlyBatch,
        target_tokens: OneOrMoreTokenSequences,
        target_ids: TargetIdsTensor,
        contrast_batch: Optional[DecoderOnlyBatch] = None,
        contrast_targets_alignments: Optional[list[list[tuple[int, int]]]] = None,
    ) -> FeatureAttributionStepOutput:
        """
        Adds to the step_output object additional information. It is called after each step.
        In our case it simply adds the prefix and (i.e. input + generated words so far) and the target (i.e. generated token at this step.)
        """

        r"""Enriches the attribution output with token information, producing the finished
        :class:`~inseq.data.FeatureAttributionStepOutput` object.

        Args:
            step_output (:class:`~inseq.data.FeatureAttributionStepOutput`): The output produced
                by the attribution step, with missing batch information.
            batch (:class:`~inseq.data.DecoderOnlyBatch`): The batch on which attribution was performed.
            target_ids (:obj:`torch.Tensor`): Target token ids of size `(batch_size, 1)` corresponding to tokens
                for which the attribution step was performed.

        Returns:
            :class:`~inseq.data.FeatureAttributionStepOutput`: The enriched attribution output.
        """
        # print(f"Calling enrich_step_output (decoder only)")
        if target_ids.ndim == 0:
            target_ids = target_ids.unsqueeze(0)
        step_output.source = None
        if contrast_batch is not None:
            contrast_aligned_idx = get_aligned_idx(len(batch.target_tokens[0]), contrast_targets_alignments[0])
            contrast_target_ids = contrast_batch.target_ids[:, contrast_aligned_idx]
            step_output.target = join_token_ids(
                tokens=target_tokens,
                ids=attribution_model.convert_ids_to_tokens(contrast_target_ids, skip_special_tokens=False),
                contrast_tokens=attribution_model.convert_ids_to_tokens(
                    contrast_target_ids[None, ...], skip_special_tokens=False
                ),
            )
            step_output.prefix = join_token_ids(tokens=batch.target_tokens, ids=batch.target_ids.tolist())
        else:
            step_output.target = join_token_ids(target_tokens, [[idx] for idx in target_ids.tolist()])
            step_output.prefix = join_token_ids(batch.target_tokens, batch.target_ids.tolist())
        return step_output
    
    # For now left the same as decoder one
    @staticmethod
    def format_forward_args(forward_fn: ForwardMethod) -> Callable[..., CustomForwardOutput]:
        @wraps(forward_fn)
        def formatted_forward_input_wrapper(
            self: "VLMAttributionModel",
            forward_tensor: AttributionForwardInputs,
            input_ids: IdsTensor,
            target_ids: ExpandedTargetIdsTensor,
            attributed_fn: Callable[..., SingleScorePerStepTensor],
            attention_mask: Optional[IdsTensor] = None,
            use_embeddings: bool = True,
            attributed_fn_argnames: Optional[list[str]] = None,
            *args,
            **kwargs,
        ) -> CustomForwardOutput:
            batch = self.formatter.convert_args_to_batch(
                decoder_input_ids=input_ids,
                decoder_attention_mask=attention_mask,
                decoder_input_embeds=forward_tensor if use_embeddings else None,
            )
            print(f"Batch is:\n{batch}")
            print(f"target_ids: {target_ids}")
            print(f"*args: {args}")    # Contains contrastive generation (contextless one) and alignments (problem since one too many!)
                                       #  VLM CCI: *args: (None, 'Describe this image\nun', [[(261, 261), (262, 262)]])
            print(f"kwargs: {kwargs}") #  VLM CCI: kwargs: {}
            print(f"Forwoard_fn: {forward_fn}") #
            return forward_fn(
                self, batch, target_ids, attributed_fn, use_embeddings, attributed_fn_argnames, *args, **kwargs
            )

        return formatted_forward_input_wrapper

 



   

   
    
class VLMAttributionModel(AttributionModel):
    """AttributionModel class for attributing VLM models."""

    formatter = VLMInputFormatter
    def get_forward_output(
        self,
        batch: DecoderOnlyBatch,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        import torch
        # print(f"Calling forward with use_embeddings: {use_embeddings}")
        # print(f"To generate output input embeds are:\n{batch.input_embeds[0, 5, :10]}")
        # For VLM we call directly the language model with the specified embeddings
        #print(f"Calling forward with input_embeds: {use_embeddings}") # Should be true
        #print(f"Batch attention mask has shape: {batch.attention_mask.shape}")
        # Save embeddings for examination
        # TODO: Investigate why but we need to check why we have a matrix of shape [1, 261] for attention while we would need here a [1, 1, 261, 261] one!
        #       For now obtain a suitable attention mask here!
        #       Also do the same for the positional_ids

        step = batch.attention_mask.shape[-1]
        expanded_mask = batch.attention_mask.view(1, 1, step, 1).expand(-1, -1, -1, step)
        final_mask = expanded_mask * expanded_mask.transpose(2, 3)
        final_mask = final_mask.float()
        positional_ids = torch.arange(1, step + 1).unsqueeze(0)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #print(f"WHOO CALLING LANGUAGE MODEL HERE with args:\n")
        #print(f"input_ids:\n{batch.input_ids.to(device) if not use_embeddings else None,}")
        #print(f"input_embeds:\n{batch.input_embeds.to(device) if use_embeddings else None,}")
        #print(f"Attention mask with shape: {final_mask.shape}")
        #print(f"position_ids with shape: {positional_ids.shape}")
        #print(f"kwargs: {kwargs}")
        return self.model.language_model( 
            input_ids=batch.input_ids.to(device) if not use_embeddings else None,
            inputs_embeds=batch.input_embeds.to(device) if use_embeddings else None,
            # Hacky fix for petals' distributed models while awaiting attention_mask support:
            # https://github.com/bigscience-workshop/petals/pull/206
            #attention_mask=batch.attention_mask if not self.is_distributed else None,
            attention_mask=final_mask.to(device),
            position_ids = positional_ids.to(device),
            **kwargs,
        )

    @formatter.format_forward_args
    def forward(self, *args, **kwargs) -> LogitsTensor:
        return self._forward(*args, **kwargs)

    @formatter.format_forward_args
    def forward_with_output(self, *args, **kwargs) -> ModelOutput:
        return self._forward_with_output(*args, **kwargs)

    def get_encoder(self) -> torch.nn.Module:
        raise NotImplementedError("Decoder-only models do not have an encoder.")

    def get_decoder(self) -> torch.nn.Module:
        return self.model.language_model
    
    def get_vision_tower(self) -> torch.nn.Module:
        return self.model.vision_tower
    
    def get_multimodal_projector(self) -> torch.nn.Module:
        return self.model.multimodal_projector
