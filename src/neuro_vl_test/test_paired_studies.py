import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import ast
import random
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from transformers import logging

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from neuro_vl_test.params import parse_args
from neuro_vl_test.prepare_paired_studies_inputs_for_generation import generate
from neuro_vl_test.distributed import is_master, init_distributed_device, all_gather_object
from neuro_vl_test.utils import (
    load_image,
    parse_output,
    set_data_device,
    set_model_bfloat16,
)

from neuro_vl.neuro_vl import VisionLanguageModel


logging.set_verbosity_error()


# utils
def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


# data
class CSVDataset(Dataset):
    def __init__(self, data, not_use_scan_types):
        self.patients = []
        assert len(data) == 1
        key, val = next(iter(data.items()))
        df = pd.read_csv(val, dtype=str, keep_default_na=False)
        for row_index, row in df.iterrows():
            current_study = os.path.join(key, str(row.get('dataset')), str(row.get('patient')), str(row.get('current_study')))
            current_study_intro = str(row.get('current_study_intro'))
            current_files = ast.literal_eval(str(row.get('current_files')))
            current_types = ast.literal_eval(str(row.get('current_types')))
            
            prior_study = os.path.join(key, str(row.get('dataset')), str(row.get('patient')), str(row.get('prior_study')))
            prior_study_intro = str(row.get('prior_study_intro'))
            prior_files = ast.literal_eval(str(row.get('prior_files')))
            prior_types = ast.literal_eval(str(row.get('prior_types')))

            questions = [str(row.get('question'))] # questions should be a list of questions.
            answers = [str(row.get('answer'))] # answers should be a list of answers.
            
            self.patients.append({
                'current_study': current_study,
                'current_study_intro': current_study_intro,
                'current_files': current_files,
                'current_types': current_types,
                'prior_study': prior_study,
                'prior_study_intro': prior_study_intro,
                'prior_files': prior_files,
                'prior_types': prior_types,
                'questions': questions,
                'answers': answers,
                'row_index': row_index
            })

        # debug
        # self.patients = self.patients[: 32]

        self.not_use_scan_types=not_use_scan_types
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        patient = self.patients[idx]
        row_index = patient['row_index']

        # get image
        current_scans = [os.path.join(patient['current_study'], file, 'img.pt') for file in patient['current_files']]
        current_image = load_image(scans=current_scans, normalizer=self.normalizer)
        prior_scans = [os.path.join(patient['prior_study'], file, 'img.pt') for file in patient['prior_files']]
        prior_image = load_image(scans=prior_scans, normalizer=self.normalizer)

        # get scan type
        current_scan_types = ['unknown'] * len(current_scans) if self.not_use_scan_types else patient['current_types']
        prior_scan_types = ['unknown'] * len(prior_scans) if self.not_use_scan_types else patient['prior_types']

        # get question and answer
        questions = patient['questions']
        answers = patient['answers']
        
        return {
            'row_index': row_index,
            'current_intros': patient['current_study_intro'],
            'current_scans': current_image, 
            'current_scan_types': current_scan_types,
            'current_num_scans': len(current_scans),
            'prior_intros': patient['prior_study_intro'],
            'prior_scans': prior_image,
            'prior_scan_types': prior_scan_types,
            'prior_num_scans': len(prior_scans),
            'questions': questions,
            'answers': answers
        }


def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in batch[0]}
    return {
        'row_index': torch.tensor(batch['row_index'], dtype=torch.long),
        'current_intros': batch['current_intros'],
        'current_scans': torch.cat(batch['current_scans'], dim=0),
        'current_scan_types': sum(batch['current_scan_types'], []),
        'current_num_scans': torch.tensor(batch['current_num_scans'], dtype=torch.long),
        'prior_intros': batch['prior_intros'],
        'prior_scans': torch.cat(batch['prior_scans'], dim=0),
        'prior_scan_types': sum(batch['prior_scan_types'], []),
        'prior_num_scans': torch.tensor(batch['prior_num_scans'], dtype=torch.long),
        'questions': batch['questions'],
        'answers': batch['answers'],
    }


# inference
def inference(model, dataloader, device, args):
    model = model.module if hasattr(model, "module") else model
    model.eval()
    results = dict()

    with torch.inference_mode():
        for batch in tqdm(dataloader, total=len(dataloader), disable=(args.rank != 0)):
            batch = set_data_device(batch, device)
            row_index = batch.pop('row_index')
            questions = batch.get('questions')
            answers = batch.pop('answers')
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                generated_ids = generate(
                    model, 
                    **batch,
                    encode_studies_independently=args.encode_studies_independently,
                    enable_thinking=args.enable_thinking,
                    num_return_sequences=args.num_samples,
                    max_new_tokens=4096, 
                    do_sample=True, 
                    temperature=args.temperature,
                    top_k=20,
                    top_p=0.95,
                    max_time=120.0
                )
                generated_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            ptr = 0
            row_index = row_index.detach().cpu().tolist()
            for idx, questions_one_batch, answers_one_batch in zip(row_index, questions, answers):
                assert len(questions_one_batch) == 1
                assert len(answers_one_batch) == 1
                samples = [
                    parse_output(generated_text[ptr + sample_idx])
                    for sample_idx in range(args.num_samples)
                ]
                results[idx] = {
                    'neuro-vl thinking': repr([sample['thinking'] for sample in samples]),
                    'neuro-vl answer': repr([sample['answer'] for sample in samples]),
                }
                ptr += args.num_samples

            assert ptr == len(generated_text), f"ptr({ptr}) != len(generated_text)({len(generated_text)})"

    return results


def main(args):
    args = parse_args(args)
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    device = init_distributed_device(args)
    if args.distributed:
        print(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        print(f'Running with a single process. Device {args.device}.')
    random_seed(args.seed, 0)

    # create model
    model = VisionLanguageModel(
        system_message=args.system_message,
        visual_resolution=args.visual_resolution,
        visual_encoder_name=args.visual_encoder_name,
        visual_encoder_path=args.visual_encoder_path,
        connector_path=args.connector_path,
        language_model_name_or_path=args.language_model_name_or_path,
        freeze_visual_encoder=True,
        freeze_connector=True,
        freeze_language_model=True,
        gradient_checkpointing=False
    )
    random_seed(args.seed, args.rank)

    # NOTE: do not use FSDP during evaluation as it only wrap the forward function
    set_model_bfloat16(model)
    model.to(device)
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], static_graph=False)

    # create dataset
    dataset = CSVDataset(data=args.valid_data, not_use_scan_types=args.not_use_scan_types)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.per_device_eval_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=sampler,
        drop_last=False
    )

    if is_master(args):
        print(f'Start Evaluation')

    local_results = inference(model, dataloader, device, args)
    if args.distributed:
        world_results = all_gather_object(args, local_results)
    else:
        world_results = [local_results]

    if is_master(args):
        results = {}
        for output in world_results:
            results.update(output)

        assert len(args.valid_data) == 1
        _, input_file = next(iter(args.valid_data.items()))
        df = pd.read_csv(input_file, dtype=str, keep_default_na=False)
        df['neuro-vl thinking'] = ''
        df['neuro-vl answer'] = ''
        for row_index, item in results.items():
            df.at[row_index, 'neuro-vl thinking'] = item['neuro-vl thinking']
            df.at[row_index, 'neuro-vl answer'] = item['neuro-vl answer']

        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        output_path = output_path / f"{args.language_model_name_or_path.split('/')[-2]}.csv"
        df.to_csv(output_path, index=False)

        print(f"[rank 0] saved results")

    return
    

if __name__ == "__main__":
    main(sys.argv[1:])
