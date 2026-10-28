import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing, checkpoint_wrapper, CheckpointImpl
)

from transformers import AutoModelForCausalLM, AutoTokenizer

from timm import create_model
from timm.layers.mlp import Mlp

from neuro_vl import visual_encoder
from neuro_vl.file_utils import pt_load


class VisionLanguageModel(nn.Module):
    def __init__(
        self, 
        system_message,
        visual_resolution,
        visual_encoder_name,
        visual_encoder_path,
        connector_path,
        language_model_name_or_path,
        freeze_visual_encoder,
        freeze_connector,
        freeze_language_model,
        gradient_checkpointing,
    ):
        super().__init__()
        self.system_message = system_message
        self.visual_resolution = visual_resolution
        self.visual_encoder_name = visual_encoder_name

        # load visual encoder
        visual_encoder = create_model(self.visual_encoder_name, pretrained=False)
        if visual_encoder_path is not None:
            ckpt = pt_load(visual_encoder_path, map_location='cpu')
            sd = ckpt['state_dict']
            trunk_sd = {k[len('module.visual.trunk.'):]: v for k, v in sd.items() if k.startswith('module.visual.trunk.')}
            sd = trunk_sd or sd
            visual_encoder.load_state_dict(sd, strict=True)
        if freeze_visual_encoder:
            for p in visual_encoder.parameters():
                p.requires_grad = False
        if gradient_checkpointing: # NOTE: previously using visual_encoder.set_grad_checkpointing(enable=True)
            from timm.models.vision_transformer import Block
            apply_activation_checkpointing(
                visual_encoder,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda m: isinstance(m, Block),
            )
        self.visual_encoder = visual_encoder

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            language_model_name_or_path,
            trust_remote_code=True,
            padding_side='left' # NOTE: this is important for generation
        )

        # load language model
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.float,
            low_cpu_mem_usage=True,
        )
        if freeze_language_model:
            for p in self.language_model.parameters():
                p.requires_grad = False
        if gradient_checkpointing: # NOTE: previously using self.language_model.gradient_checkpointing_enable()
            from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
            apply_activation_checkpointing(
                self.language_model,
                checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(m, checkpoint_impl=CheckpointImpl.NO_REENTRANT),
                check_fn=lambda m: isinstance(m, Qwen3DecoderLayer),
            )
        self.hidden_size = self.language_model.config.hidden_size

        # load connector
        # NOTE: *4 is the hard code of the merge strategy
        mlp = Mlp(in_features=self.visual_encoder.embed_dim * 4, hidden_features=self.hidden_size, out_features=self.hidden_size, use_conv=False)
        if connector_path is not None:
            ckpt = pt_load(connector_path, map_location='cpu')
            sd = ckpt['state_dict']
            sd = {(k[len('connector.'):] if 'connector.' in k else k): v for k, v in sd.items()}
            mlp.load_state_dict(sd)
        if freeze_connector:
            for p in mlp.parameters():
                p.requires_grad = False
        self.connector = mlp

    def forward_vision(self, scans, num_scans):
        # forward visual encoder 
        feature_output = self.visual_encoder.forward_intermediates(
            scans, num_scans, 
            feature_idx=[23] if 'vit_large' in self.visual_encoder_name else [11], # NOTE: hard code the visual encoder size and feature_idx
            feature_output_fmt='NDHWC',
        )

        # NOTE: hard code the merge strategy and the granularity
        features = next(iter(feature_output.values()))
        N, D, H, W, C = features.shape
        features = features.reshape(N, D, H//2, 2, W//2, 2, C).permute(0, 1, 2, 4, 3, 5, 6).flatten(4)
        return features
    
    def forward_connector(self, features):
        features = self.connector(features)
        if features.shape[1] != self.visual_resolution[0] or features.shape[2] != self.visual_resolution[1] or features.shape[3] != self.visual_resolution[2]:
            orig_dtype = features.dtype
            features = torch.nn.functional.interpolate(features.permute(0, 4, 1, 2, 3).float(), size=self.visual_resolution, mode='trilinear').permute(0, 2, 3, 4, 1)
            features = features.to(dtype=orig_dtype)
        return features

    def forward(
        self, 
        scans, num_scans,
        input_ids, labels, attention_mask,
        **kwargs
    ):
        # forward vision
        features = self.forward_vision(scans, num_scans)
        features = self.forward_connector(features)
        features = features.to(dtype=torch.bfloat16)
        
        # flatten visual tokens
        visual_tokens = features.reshape(-1, self.hidden_size)

        # prepare input embeddings for language model
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
        image_pad_mask = (input_ids == 151655) # NOTE: 151655 is "<|image_pad|>"
        inputs_embeds[image_pad_mask] = visual_tokens

        # forward language
        outputs = self.language_model(
            use_cache=False,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        return outputs
