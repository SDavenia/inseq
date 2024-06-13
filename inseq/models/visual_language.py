import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

import torch
import torchvision.transforms as transforms


from ..attr.feat import join_token_ids
from ..attr.step_functions import StepFunctionDecoderOnlyArgs
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
        inputs: FeatureAttributionInput, # FOR VLM INPUTS IS A TUBLE (str, image)
        include_eos_baseline: bool = False,
        skip_special_tokens: bool = False,
    ) -> DecoderOnlyBatch:
        """ 
        Prepares the input for attribution, i.e. it prepares a DecoderOnlyBatch object from the input.
        This contains:
            - encoding: a BatchEncodingObject (input_ids, attention_mask, input_tokens, baseline_ids, pixel_values)
            - embedding: a BatchEmbeddingObject (input_embeds, baseline_embeds, black_image_embeds).
        """
        print(f"Calling prepare_inputs_for_attributions (VLM Model)")
        #print(f"Inputs are: {inputs}")
        batch = get_batch_from_inputs(
            attribution_model,
            inputs=inputs, # To be called here inputs should be (textual_input, context_image)
            include_eos_baseline=include_eos_baseline,
            as_targets=False,
            skip_special_tokens=skip_special_tokens,
        )
        # print(f"Returning\n{DecoderOnlyBatch.from_batch(batch)}")
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
        return [f.name for f in StepFunctionDecoderOnlyArgs.__dataclass_fields__.values()]
    
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
        print(f"Calling format_attribution_args (decoder only)")
        if attribute_batch_ids:
            inputs = (batch.input_ids,)
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
    ) -> StepFunctionDecoderOnlyArgs:
        return StepFunctionDecoderOnlyArgs(
            attribution_model=attribution_model,
            forward_output=forward_output,
            target_ids=target_ids,
            is_attributed_fn=is_attributed_fn,
            decoder_input_ids=batch.target_ids,
            decoder_attention_mask=batch.target_mask,
            decoder_input_embeds=batch.target_embeds,
            context_image=transforms.ToPILImage(mode='RGB')(batch.encoding.pixel_values[0])
        )

    # LEFT THE SAME AS DECODER ONLY ARGS: AS WE WILL HAVE THEM WORKING IN THE SAME WAY
    @staticmethod
    def convert_args_to_batch(
        args: StepFunctionDecoderOnlyArgs = None,
        decoder_input_ids: Optional[IdsTensor] = None,
        decoder_attention_mask: Optional[IdsTensor] = None,
        decoder_input_embeds: Optional[EmbeddingsTensor] = None,
        **kwargs,
    ) -> DecoderOnlyBatch:
        print(f"Calling convert_args_to_batch (decoder only)")
        if args is not None:
            decoder_input_ids = args.decoder_input_ids
            decoder_attention_mask = args.decoder_attention_mask
            decoder_input_embeds = args.decoder_input_embeds
        encoding = BatchEncoding(decoder_input_ids, decoder_attention_mask)
        embedding = BatchEmbedding(decoder_input_embeds)
        return DecoderOnlyBatch(encoding, embedding)



   

   
    
class VLMAttributionModel(AttributionModel):
    """AttributionModel class for attributing VLM models."""

    formatter = VLMInputFormatter

    def get_forward_output(
        self,
        batch: DecoderOnlyBatch,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        print(f"Calling forward with use_embeddings: {use_embeddings}")
        # For VLM we call directly the language model with the specified embeddings
        return self.model.language_model( # self.model.language_model would be what we want to do here. 
            input_ids=batch.input_ids if not use_embeddings else None,
            inputs_embeds=batch.input_embeds if use_embeddings else None,
            # Hacky fix for petals' distributed models while awaiting attention_mask support:
            # https://github.com/bigscience-workshop/petals/pull/206
            attention_mask=batch.attention_mask if not self.is_distributed else None,
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
