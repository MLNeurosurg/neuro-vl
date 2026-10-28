import os
import sys
sys.path.append(os.path.abspath('.'))

import ast
import json
import math
import random
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, ConcatDataset, random_split
from torchvision.transforms import Normalize

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from neuro_vl_sft.metadata import *


def load_image(scans, normalizer):
    image = []
    for scan in scans:
        img = torch.load(scan, weights_only=True)
        img = img.float() / 255.0
        img = normalizer(img[None, ...]).squeeze()
        image.append(img[None, ...])
    return torch.stack(image, dim=0).to(dtype=torch.bfloat16) # NOTE: convert image to torch.bfloat16 by default


def process_scan_name_mr(scan_str):
    chunks = [chunk.strip() for chunk in scan_str.split(';')]
    _, orientation, region, base, suppression, geometry, contrast = chunks
    orientation = orientation.lower()
    region = region.split(',')[0].strip().lower()
    suppression = suppression.upper()
    geometry = geometry.lower()
    contrast = contrast.lower()

    # display orientation
    orientation_display = None if orientation == 'none' else orientation
    
    # display region
    region_display = MR_REGION_DISPLAY_NAMES.get(region, region.replace('_', ' '))

    # display base
    base_display = MR_BASE_DISPLAY_NAMES.get(base)
    if base_display is None: # the base is unknown
        return orientation_display, region_display, None
    if base in {'localizer', 'bolus'}:
        return None, None, base_display

    # add suppression
    suppression_labels = {label.strip().upper() for label in suppression.split(',') if label.strip()}
    if base == 'T2' and 'FLAIR' in suppression_labels:
        base_display = f"{base_display} FLAIR"
    if 'STIR' in suppression_labels:
        base_display = f"{base_display} STIR"
    if base in {'T1', 'T2'} and 'FAT_SAT' in suppression_labels:
        base_display = f"{base_display} fat-sat"

    # add contrast
    if contrast == 'with_contrast':
        if base == 'T1':
            base_display = f"post-contrast {base_display}"
        elif base == 'T2' and 'FLAIR' in suppression_labels:
            base_display = f"post-contrast {base_display}"
        elif base in {'angiography', 'venography'}:
            base_display = f"contrast-enhanced {base_display}"

    # add geometry
    if geometry == 'mip':
        base_display = f"{base_display} MIP"

    return orientation_display, region_display, base_display


def process_scan_name_ct(scan_str):
    chunks = [chunk.strip() for chunk in scan_str.split(';')]
    _, orientation, region, base, window, geometry, contrast = chunks
    orientation = orientation.lower()
    region = region.split(',')[0].strip().lower()
    base = base.lower()
    window = window.lower()
    geometry = geometry.lower()
    contrast = contrast.lower()

    # display orientation
    orientation_display = None if orientation == 'none' else orientation

    # display region
    region_display = CT_REGION_DISPLAY_NAMES.get(region, region.replace('_', ' '))

    # display base
    base_display = CT_BASE_DISPLAY_NAMES.get(base)
    if base_display is None: # the base is unknown
        return orientation_display, region_display, None
    if base in {'localizer', 'bolus'}:
        return None, None, base_display

    # add window
    if base == 'standard':
        window_display = CT_WINDOW_DISPLAY_NAMES.get(window, window.replace('_', ' '))
        if window_display is None:
            return orientation_display, region_display, None
        base_display = f"{base_display} {window_display} window"

    # display contrast
    if contrast == 'with_contrast' and base == 'standard':
        base_display = f"post-contrast {base_display}"

    # display geometry
    if geometry == 'mip':
        base_display = f"{base_display} MIP"

    return orientation_display, region_display, base_display


def system_message(system_message, tokenizer):
    input_ids = tokenizer(
        f"<|im_start|>system\n{system_message}<|im_end|>\n",
        add_special_tokens=False
    ).input_ids
    labels = [-100] * len(input_ids)
    return input_ids, labels


def add_visual_tokens(
    input_ids, labels, tokenizer,
    study_introduction, 
    scan_tags, scan_types,
    num_scans, num_visual_tokens_per_scan,
):
    if scan_tags and scan_types:
        visual_text = ''.join(
            (
                f"Scan {scan_tag} ({scan_type}): "
                if scan_type != 'unknown'
                else f"Scan {scan_tag}: "
            ) + 
            f"<|vision_start|>{'<|image_pad|>' * num_visual_tokens_per_scan}<|vision_end|>" 
            for scan_tag, scan_type in zip(scan_tags, scan_types)
        )
    else:
        visual_text = num_scans * f"<|vision_start|>{'<|image_pad|>' * num_visual_tokens_per_scan}<|vision_end|>"

    text = "<|im_start|>user\n" + study_introduction + visual_text + "<|im_end|>\n"
    _ids = tokenizer(text, add_special_tokens=False).input_ids
    input_ids.extend(_ids)
    labels.extend([-100] * len(_ids))
    return input_ids, labels


def add_user_content(content, input_ids, labels, tokenizer):
    _ids = tokenizer(
        f"<|im_start|>user\n{content}<|im_end|>\n", 
        add_special_tokens=False
    ).input_ids
    input_ids.extend(_ids)
    labels.extend([-100] * len(_ids))
    return input_ids, labels


def add_assistant_content(thinking, answer, input_ids, labels, tokenizer):
    if thinking:
        _ids = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
    else:
        _ids = tokenizer("<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False).input_ids
    input_ids.extend(_ids)
    labels.extend([-100] * len(_ids))

    if thinking:
        _ids = tokenizer(f"<think>\n{thinking}\n</think>\n\n{answer}<|im_end|>\n", add_special_tokens=False).input_ids
    else:
        _ids = tokenizer(f"{answer}<|im_end|>\n", add_special_tokens=False).input_ids
    input_ids.extend(_ids)
    labels.extend(_ids)

    return input_ids, labels


class PreTrainDataset(Dataset):
    def __init__(
        self,
        data, # {root: study.txt}
        max_num_scans=8,
        visual_resolution=(8, 7, 7),
        system_message='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.',
        context_length=4096,
        tokenizer=None,
        is_train=False,
    ):
        self.studies = []
        for key, val in data.items():
            with open(val, 'r', encoding='utf-8') as f:
                uids = [line.strip() for line in f if line.strip()]
            for uid in uids:
                self.studies.append(os.path.join(key, uid))

        # debug
        # self.studies = self.studies[:1024]

        self.max_num_scans = max_num_scans
        self.visual_resolution = visual_resolution
        self.system_message = system_message
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study = self.studies[idx]
        with open(os.path.join(study, 'pre_train.json'), 'r', encoding='utf-8') as f:
            info = json.load(f)

        # get image
        scans = info['scans']
        if self.is_train:
            random.shuffle(scans)
        scans = scans[:self.max_num_scans] # NOTE: specify this to avoid OOM during evaluation
        scans = [os.path.join(study, scan, 'img.pt') for scan in scans]
        image = load_image(scans=scans, normalizer=self.normalizer)

        # get report
        reports = info['reports']
        if self.is_train:
            report = random.choice(reports)
        else:
            report = reports[0]
        report = ' '.join(report)

        # process
        input_ids, labels = system_message(self.system_message, self.tokenizer)
        input_ids, labels = add_visual_tokens(
            input_ids=input_ids,
            labels=labels,
            tokenizer=self.tokenizer,
            study_introduction='',
            scan_tags=[],
            scan_types=[],
            num_scans=len(scans),
            num_visual_tokens_per_scan=math.prod(self.visual_resolution)
        )
        input_ids, labels = add_assistant_content(
            thinking='',
            answer=report,
            input_ids=input_ids,
            labels=labels,
            tokenizer=self.tokenizer
        )

        # to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # truncate
        input_ids = input_ids[:self.context_length]
        labels = labels[:self.context_length]
        attention_mask = attention_mask[:self.context_length]

        # padding
        input_ids = F.pad(input_ids, (self.context_length - input_ids.shape[0], 0), value=self.tokenizer.pad_token_id)
        labels = F.pad(labels, (self.context_length - labels.shape[0], 0), value=-100)
        attention_mask = F.pad(attention_mask, (self.context_length - attention_mask.shape[0], 0), value=0)

        return {
            'scans': image,
            'num_scans': len(scans),
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class SingleStudyDataset(Dataset):
    def __init__(
        self,
        data, # NOTE: {root: [study.txt] or 'study.txt'}
        visual_resolution=(8, 7, 7),
        not_use_scan_types=False, # NOTE: for ablation study only
        use_confession_statement=False, # NOTE: departure
        enable_thinking=False,
        system_message='You are Neuro-VL, created by MLiNS @ University of Michigan. You are a helpful assistant.',
        context_length=5632,
        tokenizer=None,
        is_train=True,
    ):
        self.studies = []
        for key, val in data.items():
            if isinstance(val, str):
                val = [val]
            for v in val:
                with open(v, 'r', encoding='utf-8') as f:
                    data = [tuple(part.strip() for part in line.strip().split(',')) for line in f if line.strip()]
                for uid, conv in data:
                    self.studies.append((os.path.join(key, uid), conv))

        # debug
        # self.studies = self.studies[:512]

        self.visual_resolution = visual_resolution
        self.not_use_scan_types = not_use_scan_types
        self.use_confession_statement = use_confession_statement
        self.enable_thinking = enable_thinking
        self.system_message = system_message
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, idx):
        study, conv = self.studies[idx]
        with open(os.path.join(study, f'{conv}.json'), 'r', encoding='utf-8') as f:
            info = json.load(f)
        scans = info['scans']
        normalized_scans = info['scans_normalized']
        scan_tags = info["scans_tag"] if self.is_train else [i + 1 for i in range(len(scans))]

        # get image
        image = load_image(scans=[os.path.join(study, scan, 'img.pt') for scan in scans], normalizer=self.normalizer)

        # get scan_types and scan_statement
        scan_types = []
        scan_statements = []
        if not self.is_train:
            is_provided = [True] * len(normalized_scans)
        elif np.random.rand() < 0.5:
            is_provided = (np.random.permutation(len(normalized_scans)) < np.random.randint(len(normalized_scans) + 1)).tolist()
        else:
            is_provided = [False] * len(normalized_scans)

        for scan_tag, provid_scan, normalized_scan in zip(scan_tags, is_provided, normalized_scans):
            modality = normalized_scan.split(';')[0].strip().upper()
            if modality == 'MRI':
                orientation, region, base = process_scan_name_mr(normalized_scan)
            if modality == 'CT':
                orientation, region, base = process_scan_name_ct(normalized_scan)

            if provid_scan and base:
                scan_types.append(' '.join(part for part in (orientation, region, base) if part is not None))
                scan_statements.append(f"Scan {scan_tag}: The scan type is provided as {' '.join(part for part in (orientation, region, base) if part is not None)}.")
            elif not provid_scan and base:
                scan_types.append('unknown')
                scan_statements.append(f"Scan {scan_tag}: The scan type is predicted as {' '.join(part for part in (orientation, region, base) if part is not None)}.")
            else:
                scan_types.append('unknown')
                scan_statements.append(f"Scan {scan_tag}: The exact scan type is uncertain, it can only be predicted as {' '.join(part for part in (orientation, region, modality) if part is not None)}.")
        scan_statement = ' '.join(scan_statements)

        # get questions, thinkings, and answers
        questions, thinkings, answers = [], [], []
        if conv == 'dialogue': # dialogue question
            questions.extend(info['questions'])
            thinkings.extend(info['thinkings'])
            answers.extend(info['answers'])
        else: # other questions
            use_confession_statement = (
                self.use_confession_statement 
                and 'confession_thinking' in info 
                and 'confession_answer' in info
            )
            candidate_count = min(2, len(info["questions"])) if use_confession_statement else len(info["questions"])            
            index = np.random.randint(candidate_count) if self.is_train else 0
            questions.append(info['questions'][index])

            thinking = info['thinkings'][index]
            answer = info['answers'][index]
            if use_confession_statement:
                thinking += ('' if '\n\n[Reasoning]\n' in info['confession_thinking'] else ' ') + info['confession_thinking']
                answer += ' ' + info['confession_answer']
            thinkings.append(thinking)
            answers.append(answer)

        if self.not_use_scan_types and self.is_train:
            scan_types = ['unknown'] * len(scans)
        thinkings = [thinking.replace('Detected images: {INPUT_SCAN_NAMES}\n', '') if self.not_use_scan_types else thinking.replace('{INPUT_SCAN_NAMES}', scan_statement) for thinking in thinkings]

        # process
        input_ids, labels = system_message(self.system_message, self.tokenizer)
        input_ids, labels = add_visual_tokens(
            input_ids=input_ids,
            labels=labels,
            tokenizer=self.tokenizer,
            study_introduction=STUDY_INTRODUCTION[np.random.randint(len(STUDY_INTRODUCTION))] if self.is_train and np.random.rand() < 0.5 else '',
            scan_tags=scan_tags,
            scan_types=scan_types,
            num_scans=len(scans),
            num_visual_tokens_per_scan=math.prod(self.visual_resolution)
        )

        for question, thinking, answer in zip(questions, thinkings, answers):
            input_ids, labels = add_user_content(
                content=question,
                input_ids=input_ids,
                labels=labels,
                tokenizer=self.tokenizer
            )
            input_ids, labels = add_assistant_content(
                thinking=thinking if self.enable_thinking else '',
                answer=answer,
                input_ids=input_ids,
                labels=labels,
                tokenizer=self.tokenizer
            )

        # to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # truncate
        input_ids = input_ids[:self.context_length]
        labels = labels[:self.context_length]
        attention_mask = attention_mask[:self.context_length]

        # padding
        input_ids = F.pad(input_ids, (self.context_length - input_ids.shape[0], 0), value=self.tokenizer.pad_token_id)
        labels = F.pad(labels, (self.context_length - labels.shape[0], 0), value=-100)
        attention_mask = F.pad(attention_mask, (self.context_length - attention_mask.shape[0], 0), value=0)

        return {
            'scans': image,
            'num_scans': len(scans),
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class MultipleStudiesDataset(Dataset):
    def __init__(
        self,
        data, # NOTE: {root, [study.csv] or 'study.csv'}
        visual_resolution=(8, 7, 7),
        encode_studies_independently=False,
        enable_thinking=False,
        system_message='You are Neuro-VL, created by MLiNS @ University of Michigan. You are a helpful assistant.',
        context_length=8192,
        tokenizer=None,
        is_train=True,
    ):
        self.studies = []
        for key, val in data.items():
            if isinstance(val, str):
                val = [val]
            for v in val:
                df = pd.read_csv(v, dtype=str, keep_default_na=False)
                for _, row in df.iterrows():
                    self.studies.append({
                        'current_study_in_the_front': int(row.get('current_order')) < int(row.get('prior_order')),
                        'current_study_intro': row.get('current_intro'),
                        'current_study_path': os.path.join(key, row.get('current_modality'), row.get('current_study')),
                        'current_study_scans': ast.literal_eval(str(row.get('current_scans'))),
                        'current_study_normalized_scans': ast.literal_eval(str(row.get('current_normalized_scans'))),
                        'current_study_scan_tags': ast.literal_eval(str(row.get('current_scan_tags'))),
                        'prior_study_report': row.get("prior_report") if 'report' in row.get('prior_type') else None,
                        'prior_study_intro': row.get("prior_intro") if 'img' in row.get('prior_type') else None,
                        'prior_study_path': os.path.join(key, row.get('prior_modality'), row.get('prior_study')) if 'img' in row.get('prior_type') else None,
                        'prior_study_scans': ast.literal_eval(str(row.get('prior_scans'))) if 'img' in row.get('prior_type') else None,
                        'prior_study_normalized_scans': ast.literal_eval(str(row.get('prior_normalized_scans'))) if 'img' in row.get('prior_type') else None,
                        'prior_study_scan_tags': ast.literal_eval(str(row.get('prior_scan_tags'))) if 'img' in row.get('prior_type') else None,
                        'questions': ast.literal_eval(str(row.get('questions'))),
                        'thinkings': ast.literal_eval(str(row.get('thinkings'))),
                        'answers': ast.literal_eval(str(row.get('answers'))),
                    })
        
        # debug
        # self.studies = self.studies[:512]

        self.visual_resolution = visual_resolution
        self.encode_studies_independently = encode_studies_independently
        self.enable_thinking = enable_thinking
        self.system_message = system_message
        self.context_length = context_length
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    def __len__(self):
        return len(self.studies)

    def _process_scan_types(self, scan_tags, normalized_scans):
        scan_types = []
        scan_statements = []
        if not self.is_train:
            is_provided = [True] * len(normalized_scans)
        elif np.random.rand() < 0.5:
            is_provided = (np.random.permutation(len(normalized_scans)) < np.random.randint(len(normalized_scans) + 1)).tolist()
        else:
            is_provided = [False] * len(normalized_scans)

        for scan_tag, provid_scan, normalized_scan in zip(scan_tags, is_provided, normalized_scans):
            modality = normalized_scan.split(';')[0].strip().upper()
            if modality == 'MRI':
                orientation, region, base = process_scan_name_mr(normalized_scan)
            if modality == 'CT':
                orientation, region, base = process_scan_name_ct(normalized_scan)

            if provid_scan and base:
                scan_types.append(' '.join(part for part in (orientation, region, base) if part is not None))
                scan_statements.append(f"Scan {scan_tag}: The scan type is provided as {' '.join(part for part in (orientation, region, base) if part is not None)}.")
            elif not provid_scan and base:
                scan_types.append('unknown')
                scan_statements.append(f"Scan {scan_tag}: The scan type is predicted as {' '.join(part for part in (orientation, region, base) if part is not None)}.")
            else:
                scan_types.append('unknown')
                scan_statements.append(f"Scan {scan_tag}: The exact scan type is uncertain, it can only be predicted as {' '.join(part for part in (orientation, region, modality) if part is not None)}.")
        return scan_types, ' '.join(scan_statements)

    def __getitem__(self, idx):
        study = self.studies[idx]

        # get image
        current_study_image = load_image([os.path.join(study['current_study_path'], scan, 'img.pt') for scan in study['current_study_scans']], normalizer=self.normalizer)
        prior_study_image = load_image([os.path.join(study['prior_study_path'], scan, 'img.pt') for scan in study['prior_study_scans']], normalizer=self.normalizer) if study['prior_study_path'] is not None else None
        if prior_study_image is None:
            image = current_study_image
        elif study['current_study_in_the_front']:
            image = torch.cat([current_study_image, prior_study_image], dim=0)
        else:
            image = torch.cat([prior_study_image, current_study_image], dim=0)

        # get scan_types and scan_statement
        current_study_scan_types, current_study_scan_statement = self._process_scan_types(
            scan_tags=study['current_study_scan_tags'],
            normalized_scans=study['current_study_normalized_scans'],
        )
        (prior_study_scan_types, prior_study_scan_statement) = self._process_scan_types(
            scan_tags=study['prior_study_scan_tags'],
            normalized_scans=study['prior_study_normalized_scans'],
        ) if study['prior_study_path'] is not None else (None, '')

        # get prior report
        prior_study_report = study['prior_study_report']

        # get questions, thinkings, and answers
        # NOTE: the csv file has been sampled, therefore, no subquestion sampling is needed during training
        questions, thinkings, answers = [], [], []
        questions.extend(study['questions'])
        thinkings.extend(study['thinkings'])
        answers.extend(study['answers'])

        thinkings = [thinking.replace('{CURRENT_INPUT_SCAN_NAMES}', current_study_scan_statement) for thinking in thinkings]
        thinkings = [thinking.replace('{PRIOR_INPUT_SCAN_NAMES}', prior_study_scan_statement) for thinking in thinkings]

        if self.is_train and prior_study_report is not None and np.random.rand() < 0.5:
            questions[0] = f"{prior_study_report}\n\n{questions[0]}"
            prior_study_report = None

        # process
        input_ids, labels = system_message(self.system_message, self.tokenizer)

        current_study_input_ids, current_study_labels = add_visual_tokens(
            input_ids=[],
            labels=[],
            tokenizer=self.tokenizer,
            study_introduction=study['current_study_intro'] + ' ',
            scan_tags=study['current_study_scan_tags'],
            scan_types=current_study_scan_types,
            num_scans=len(study['current_study_scans']),
            num_visual_tokens_per_scan=math.prod(self.visual_resolution)
        )

        if prior_study_image is not None and prior_study_report is None:
            prior_study_input_ids, prior_study_labels = add_visual_tokens(
                input_ids=[],
                labels=[],
                tokenizer=self.tokenizer,
                study_introduction=study['prior_study_intro'] + ' ',
                scan_tags=study['prior_study_scan_tags'],
                scan_types=prior_study_scan_types,
                num_scans=len(study['prior_study_scans']),
                num_visual_tokens_per_scan=math.prod(self.visual_resolution)
            )
        elif prior_study_image is None and prior_study_report is not None:
            prior_study_input_ids, prior_study_labels = add_user_content(
                content=prior_study_report,
                input_ids=[],
                labels=[],
                tokenizer=self.tokenizer
            )
        elif prior_study_image is not None and prior_study_report is not None:
            prior_study_input_ids, prior_study_labels = add_visual_tokens(
                input_ids=[],
                labels=[],
                tokenizer=self.tokenizer,
                study_introduction=study['prior_study_intro'] + ' ',
                scan_tags=study['prior_study_scan_tags'],
                scan_types=prior_study_scan_types,
                num_scans=len(study['prior_study_scans']),
                num_visual_tokens_per_scan=math.prod(self.visual_resolution)
            )
            prior_study_input_ids, prior_study_labels = add_user_content(
                content=prior_study_report,
                input_ids=prior_study_input_ids,
                labels=prior_study_labels,
                tokenizer=self.tokenizer
            )
        else:
            prior_study_input_ids, prior_study_labels = [], []

        if study['current_study_in_the_front']:
            input_ids += current_study_input_ids + prior_study_input_ids
            labels += current_study_labels + prior_study_labels
        else:
            input_ids += prior_study_input_ids + current_study_input_ids
            labels += prior_study_labels + current_study_labels

        for question, thinking, answer in zip(questions, thinkings, answers):
            input_ids, labels = add_user_content(
                content=question,
                input_ids=input_ids,
                labels=labels,
                tokenizer=self.tokenizer
            )
            input_ids, labels = add_assistant_content(
                thinking=thinking if self.enable_thinking else '',
                answer=answer,
                input_ids=input_ids,
                labels=labels,
                tokenizer=self.tokenizer
            )

        # to tensor
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        # truncate
        input_ids = input_ids[:self.context_length]
        labels = labels[:self.context_length]
        attention_mask = attention_mask[:self.context_length]

        # padding
        input_ids = F.pad(input_ids, (self.context_length - input_ids.shape[0], 0), value=self.tokenizer.pad_token_id)
        labels = F.pad(labels, (self.context_length - labels.shape[0], 0), value=-100)
        attention_mask = F.pad(attention_mask, (self.context_length - attention_mask.shape[0], 0), value=0)

        if not self.encode_studies_independently:
            num_scans = (
                len(study["current_study_scans"])
                if study["prior_study_scans"] is None
                else len(study["current_study_scans"])
                    + len(study["prior_study_scans"])
            )
        elif study["prior_study_scans"] is None:
            num_scans = len(study["current_study_scans"])
        elif study["current_study_in_the_front"]:
            num_scans = [len(study["current_study_scans"]), len(study["prior_study_scans"])]
        else:
            num_scans = [len(study["prior_study_scans"]),len(study["current_study_scans"])]

        return {
            'scans': image,
            'num_scans': num_scans,
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def simple_collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in batch[0]}
    num_scans = []
    for value in batch['num_scans']:
        if isinstance(value, list):
            num_scans.extend(value)
        else:
            num_scans.append(value)

    return {
        'scans': torch.cat(batch['scans'], dim=0),
        'num_scans': torch.tensor(num_scans, dtype=torch.long),
        'input_ids': torch.stack(batch['input_ids'], dim=0),
        'labels': torch.stack(batch['labels'], dim=0),
        'attention_mask': torch.stack(batch['attention_mask'], dim=0)
    }


def build_dataset(tokenizer, is_train, args):
    if args.stage == 'stage-one':
        dataset = PreTrainDataset(
            data=args.train_data if is_train else args.valid_data,
            max_num_scans=args.max_num_scans if is_train else args.max_num_scans * 4, # NOTE: Specify this to avoid OOM during validation
            visual_resolution=(8, 7, 7),
            system_message=args.system_message,
            context_length=args.context_length if is_train else args.context_length * 4, # NOTE: Specify this to avoid OOM during validation
            tokenizer=tokenizer,
            is_train=is_train,
        )
    if args.stage == 'stage-two':
        dataset = SingleStudyDataset(
            data=args.train_data if is_train else args.valid_data,
            visual_resolution=(8, 7, 7),
            not_use_scan_types=args.not_use_scan_types, # NOTE: for ablation study only
            use_confession_statement=args.use_confession_statement,
            enable_thinking=args.enable_thinking,
            system_message=args.system_message,
            context_length=args.context_length,
            tokenizer=tokenizer,
            is_train=is_train,
        )
    if args.stage == 'stage-three':
        train_data_txt = {key: val for key, val in args.train_data.items() if str(val).lower().endswith('.txt')}
        train_data_csv = {key: val for key, val in args.train_data.items() if str(val).lower().endswith('.csv')}
        valid_data_txt = {key: val for key, val in args.valid_data.items() if str(val).lower().endswith('.txt')}
        valid_data_csv = {key: val for key, val in args.valid_data.items() if str(val).lower().endswith('.csv')}

        single_study_dataset = SingleStudyDataset(
            data=train_data_txt if is_train else valid_data_txt,
            visual_resolution=(8, 7, 7),
            not_use_scan_types=args.not_use_scan_types, # NOTE: for ablation study only
            use_confession_statement=args.use_confession_statement,
            enable_thinking=args.enable_thinking,
            system_message=args.system_message,
            context_length=args.context_length,
            tokenizer=tokenizer,
            is_train=is_train,
        )
        multiple_studies_dataset = MultipleStudiesDataset(
            data=train_data_csv if is_train else valid_data_csv,
            visual_resolution=(8, 7, 7),
            encode_studies_independently=args.encode_studies_independently,
            enable_thinking=args.enable_thinking,
            system_message=args.system_message,
            context_length=args.context_length,
            tokenizer=tokenizer,
            is_train=is_train,
        )
        
        num_single_study = len(single_study_dataset)
        num_multiple_studies = len(multiple_studies_dataset)
        assert num_multiple_studies <= num_single_study
        single_study_dataset, _ = random_split(
            single_study_dataset,
            [num_multiple_studies, num_single_study - num_multiple_studies],
            generator=torch.Generator().manual_seed(args.seed),  # for reproducibility
        )
        dataset = ConcatDataset([single_study_dataset, multiple_studies_dataset])

    return dataset
