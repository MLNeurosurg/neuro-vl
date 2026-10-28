# NOTE: [hard code] monky patch the no_sync() in FSDP to prevent OOM when do gradient accumulation.
import contextlib, importlib, types
def _always_sync(self):
    """Replacement for .no_sync that does nothing."""
    return contextlib.nullcontext()
def disable_no_sync():
    """Monkey-patch FSDP so .no_sync returns a null-context."""
    targets = [
        ("torch.distributed.fsdp", "FullyShardedDataParallel")
    ]
    for mod_path, cls_name in targets:
        try:
            cls = getattr(importlib.import_module(mod_path), cls_name)
            cls.no_sync = types.MethodType(_always_sync, cls)
        except (ImportError, AttributeError):
            continue
disable_no_sync()


import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))
from datetime import datetime

from transformers import TrainingArguments, set_seed

from neuro_vl.neuro_vl import VisionLanguageModel
from neuro_vl_sft.data import build_dataset, simple_collate_fn
from neuro_vl_sft.params import parse_args
from neuro_vl.file_utils import pt_save_fsdp
from neuro_vl_sft.trainer import NeuroVLTrainer


def build_training_args(args):
    args.run_name = "_".join([
        f"DATE_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}",
        f"STAGE_{args.stage}",
        f"MODEL_{args.language_model_name_or_path.split('/')[-1]}"
    ])

    common_kwargs = dict(
        seed=args.seed,
        output_dir=args.output_dir,
        report_to=args.report_to,
        run_name=args.run_name,
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        dataloader_num_workers=args.dataloader_num_workers,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        tf32=args.tf32,
        eval_strategy="epoch",
        save_strategy=args.save_strategy,
        save_steps=100,
        save_total_limit=1,
        log_on_each_node=False,
        remove_unused_columns=False,
    )
    training_args = TrainingArguments(**common_kwargs)
    training_args.visual_encoder_learning_rate = args.visual_encoder_learning_rate
    training_args.freeze_visual_encoder = args.freeze_visual_encoder
    return training_args


def fsdp_auto_wrap_policy():
    import functools
    from timm.models.vision_transformer import Block
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
    from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
    
    def lambda_policy_fn(module):
        if (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        ):
            return True
        return False

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)

    # NOTE: [hard code] only compatible with "Block", "Qwen3DecoderLayer"
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(Block, Qwen3DecoderLayer),
    )

    auto_wrap_policy = functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    
    return auto_wrap_policy


def launch_training(model, train_dataset, eval_dataset, collate_fn, args):
    training_args = build_training_args(args)
    trainer = NeuroVLTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )
    
    fsdp_plugin = trainer.accelerator.state.fsdp_plugin
    fsdp_plugin.auto_wrap_policy = fsdp_auto_wrap_policy()

    # NOTE: [debug] run this line first to make sure there is no OOM during evaluation
    # trainer.evaluate()

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    pt_save_fsdp(trainer, args)
    return


def main(args):
    args = parse_args(args)
    set_seed(args.seed)

    model = VisionLanguageModel(
        system_message=args.system_message,
        visual_resolution=args.visual_resolution,
        visual_encoder_name=args.visual_encoder_name,
        visual_encoder_path=args.visual_encoder_path,
        connector_path=args.connector_path,
        language_model_name_or_path=args.language_model_name_or_path,
        freeze_visual_encoder=args.freeze_visual_encoder,
        freeze_connector=args.freeze_connector,
        freeze_language_model=args.freeze_language_model,
        gradient_checkpointing=args.gradient_checkpointing
    )

    tokenizer = model.tokenizer
    train_dataset = build_dataset(tokenizer=tokenizer, is_train=True, args=args)
    valid_dataset = build_dataset(tokenizer=tokenizer, is_train=False, args=args)

    launch_training(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        collate_fn=simple_collate_fn,
        args=args,
    )

if __name__ == "__main__":
    main(sys.argv[1:])
