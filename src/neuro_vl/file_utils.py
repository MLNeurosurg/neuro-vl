import os
import warnings
import fsspec

import torch
from torch.distributed.fsdp import FullyShardedDataParallel
from torch.distributed.fsdp.api import FullStateDictConfig, StateDictType


def pt_load(file_path, map_location=None):
    of = fsspec.open(file_path, "rb")
    with of as f:
        out = torch.load(f, map_location=map_location)
    return out


def pt_load_fsdp(model, file_path, map_location=None):
    full_state_dict = pt_load(file_path, map_location)
    load_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_dict_config=load_policy):
        model.load_state_dict(full_state_dict['state_dict'])
    return


def pt_save_fsdp(trainer, args):
    if trainer.args.local_rank != 0 and trainer.args.local_rank != -1:
        warnings.filterwarnings("ignore")

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FullyShardedDataParallel.state_dict_type(trainer.model, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
        full_state_dict = trainer.model.state_dict()

    if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        checkpoint_dir = os.path.join(trainer.args.output_dir, trainer.args.run_name)
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not args.freeze_connector:
            connector_state_dict = {k[len('connector.'):]: v for k, v in full_state_dict.items() if "connector." in k}
            connector_checkpoint = {"state_dict": connector_state_dict}
            torch.save(connector_checkpoint, os.path.join(checkpoint_dir, 'connector_checkpoint.pt'))
        
        if not args.freeze_visual_encoder:
            visual_encoder_state_dict = {k[len('visual_encoder.'):]: v for k, v in full_state_dict.items() if "visual_encoder." in k}
            visual_encoder_checkpoint = {"state_dict": visual_encoder_state_dict}
            torch.save(visual_encoder_checkpoint, os.path.join(checkpoint_dir, f'{args.visual_encoder_name}.pt'))
        
        if not args.freeze_language_model:
            language_model_state_dict = {k[len('language_model.'):]: v for k, v in full_state_dict.items() if "language_model." in k}
            trainer.model.tokenizer.save_pretrained(os.path.join(checkpoint_dir, args.language_model_name_or_path.split('/')[-1]))
            trainer.model.language_model.save_pretrained(os.path.join(checkpoint_dir, args.language_model_name_or_path.split('/')[-1]), state_dict=language_model_state_dict, safe_serialization=True)

    return
