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

    parser.add_argument("--seed", type=int, default=42)

    # data
    parser.add_argument(
        "--valid_data",
        nargs='*',
        default={},
        action=ParseKwargs,
        help="{'root': 'txt'}"
    )
    parser.add_argument(
        "--visual_resolution", nargs=3, type=int, default=(8, 7, 7),
        metavar=("D", "H", "W"), help="Spatial resolution fed to the language model."
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
        "--encode_studies_independently", action="store_true", default=False, help="Encode studies for the patients independently."
    )
    parser.add_argument(
        "--enable_thinking", action="store_true", default=False, help="Enable thinking mode in either training or evaluation."
    )
    parser.add_argument(
        "--system_message", type=str, default="You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    )

    # log
    parser.add_argument("--output_dir", type=str, default="./logs/")

    # evaluation
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=1
    )
    parser.add_argument(
        "--dataloader_num_workers", type=int, default=8
    )

    # sampling
    parser.add_argument(
        "--num_samples", type=int, default=1, help="Number of samples during sampling."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.6, help="Temperature during sampling."
    )

    return parser.parse_args(args)
