import argparse
import ast


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser(
        description="Train the Vision-Language Model"
    )

    parser.add_argument(
        "--stage", choices=["stage-one", "stage-two", "stage-three"], default="pre-train", help="Training stage."
    )
    parser.add_argument("--seed", type=int, default=42)

    # data
    parser.add_argument(
        "--train_data",
        nargs='*',
        default={},
        action=ParseKwargs,
        help="{'root': 'txt'}"
    )
    parser.add_argument(
        "--valid_data",
        nargs='*',
        default={},
        action=ParseKwargs,
        help="{'root': 'txt'}"
    )
    parser.add_argument(
        "--max_num_scans", type=int, default=8, help="Max number of scans fed to the visual encoder."
    )
    parser.add_argument(
        "--visual_resolution", nargs=3, type=int, default=(8, 7, 7),
        metavar=("D", "H", "W"), help="Spatial resolution fed to the language model."
    )
    parser.add_argument(
        "--context_length", type=int, default=4096, help="Max token length fed to the language model."
    )

    # model
    parser.add_argument(
        "--visual_encoder_name", type=str, default=None, help="Name of visual encoder."
    )
    parser.add_argument(
        "--visual_encoder_path", type=str, default=None, help="Path to the checkpoints of visual encoder."
    )
    parser.add_argument(
        "--connector_path", type=str, default=None, help="Path to the checkpoints of connector."
    )
    parser.add_argument(
        "--language_model_name_or_path", type=str, help="Any HF-compatible causal-LM checkpoint."
    )
    parser.add_argument(
        "--not_use_scan_types", action="store_true", default=False, help="Do not provide scan types supervision during training. Only turn on for ablation."
    )
    parser.add_argument(
        "--use_confession_statement", action="store_true", default=False, help="Use confession statement."
    )
    parser.add_argument(
        "--encode_studies_independently", action="store_true", default=False, help="Encode studies for the patients independently."
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=False, help="Enable thinking mode in either training or evaluation."
    )
    parser.add_argument(
        "--system_message", type=str, default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )
    parser.add_argument("--freeze_visual_encoder", action="store_true")
    parser.add_argument("--freeze_connector", action="store_true")
    parser.add_argument("--freeze_language_model", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")

    # log
    parser.add_argument("--output_dir", type=str, default="./logs/")
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument(
        "--save_strategy", type=str, choices=["no", "steps"], default="no", help="Strategy for saving resumable Trainer checkpoints."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to a Hugging Face Trainer checkpoint directory."
    )


    # train
    parser.add_argument(
        "--num_train_epochs", type=float, default=1.0
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=1
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=1
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=8
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1
    )
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument(
        "--visual_encoder_learning_rate", type=float, default=None,
        help="Learning rate for visual_encoder parameters. Connector and language_model use --learning_rate.",
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", type=str, default='cosine')
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    return parser.parse_args(args)
