import logging
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

import torch

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
        attribution_model: "DecoderOnlyAttributionModel",
        inputs: FeatureAttributionInput,
        include_eos_baseline: bool = False,
        skip_special_tokens: bool = False,
    ) -> DecoderOnlyBatch:
        """ 
        Prepares the input for attribution, i.e. it prepares a DecoderOnlyBatch object from the input.
        This contains:
            - encoding: a BatchEncodingObject (input_ids, attention_mask, input_tokens, baseline_ids, pixel_values)
            - embedding: a BatchEmbeddingObject (input_embeds, baseline_embeds)
        """
        # TODO IMPLEMENT IT
        raise NotImplementedError("VLMInputFormatter.prepare_inputs_for_attribution is not implemented yet.")
    
    # FINISCI DI PRENDERLI DA DECODER_ONLY.py

   

   
    
class VLMAttributionModel(AttributionModel):
    """AttributionModel class for attributing VLM models."""

    formatter = VLMInputFormatter

    def get_forward_output(
        self,
        batch: DecoderOnlyBatch,
        use_embeddings: bool = True,
        **kwargs,
    ) -> ModelOutput:
        return self.model( # self.model.language_model would be what we want to do here. 
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
