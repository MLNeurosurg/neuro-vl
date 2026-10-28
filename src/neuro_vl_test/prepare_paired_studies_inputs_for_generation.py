import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import math
import torch


def interleave_scans(current_scans, current_num_scans, prior_scans, prior_num_scans):
    if torch.is_tensor(current_num_scans):
        current_num_scans = current_num_scans.tolist()
    if torch.is_tensor(prior_num_scans):
        prior_num_scans = prior_num_scans.tolist()
    
    current_splits = torch.split(current_scans, current_num_scans, dim=0)
    prior_splits = torch.split(prior_scans, prior_num_scans, dim=0)

    combined_list = []
    for current, prior in zip(current_splits, prior_splits):
        # NOTE: the prior study will always be in the front of the current study
        combined_list.append(torch.cat([prior, current], dim=0)) 
    
    combined_scans = torch.cat(combined_list, dim=0)
    return combined_scans


@torch.inference_mode()
def prepare_inputs_for_generation(
    model, batched_questions, 
    batched_current_intro, current_scans, current_scan_types, current_num_scans, 
    batched_prior_intro, prior_scans, prior_scan_types, prior_num_scans,
    encode_studies_independently, enable_thinking
):
    device = current_scans.device

    # forward visual features for all studies in the batch
    scans = interleave_scans(current_scans, current_num_scans, prior_scans, prior_num_scans)
    patient_num_scans = current_num_scans + prior_num_scans

    if encode_studies_independently:
        num_scans = torch.stack(
            [prior_num_scans, current_num_scans],
            dim=1,
        ).flatten()
    else:
        num_scans = patient_num_scans

    features = model.forward_vision(scans, num_scans)
    features = model.forward_connector(features)
    features = features.to(dtype=torch.bfloat16)

    # prepare batch input
    text = []
    visual_tokens = []
    C = features.shape[-1]
    num_visual_tokens_per_scan = math.prod(model.visual_resolution)
    offset_patient = torch.cumsum(patient_num_scans, 0, dtype=torch.long) - patient_num_scans
    offset_current = torch.cumsum(current_num_scans, 0, dtype=torch.long) - current_num_scans
    offset_prior = torch.cumsum(prior_num_scans, 0, dtype=torch.long) - prior_num_scans
    for idx, questions_one_batch in enumerate(batched_questions): # questions_one_batch should be a list of questions (strings)
        visual_token = features[offset_patient[idx]:offset_patient[idx]+patient_num_scans[idx].item()].reshape(-1, C)
        visual_tokens.append(visual_token[None, ...].expand(len(questions_one_batch), -1, -1).reshape(-1, C))

        current_intro = batched_current_intro[idx]
        prior_intro=batched_prior_intro[idx]
        for question in questions_one_batch:
            system_message = f"<|im_start|>system\n{model.system_message}<|im_end|>\n"
            current_visual_text = ''.join(
                (
                    f"Scan {i + 1} ({current_scan_types[offset_current[idx] + i]}): "
                    if current_scan_types[offset_current[idx] + i] != "unknown"
                    else f"Scan {i + 1}: "
                )
                + f"<|vision_start|>{'<|image_pad|>' * num_visual_tokens_per_scan}<|vision_end|>"
                for i in range(current_num_scans[idx].item())
            )
            current_visual_text = "<|im_start|>user\n" + current_intro + ' ' + current_visual_text + "<|im_end|>\n"
            prior_visual_text = ''.join(
                (
                    f"Scan {i + 1} ({prior_scan_types[offset_prior[idx] + i]}): "
                    if prior_scan_types[offset_prior[idx] + i] != "unknown"
                    else f"Scan {i + 1}: "
                )
                + f"<|vision_start|>{'<|image_pad|>' * num_visual_tokens_per_scan}<|vision_end|>"
                for i in range(prior_num_scans[idx].item())
            )
            prior_visual_text = "<|im_start|>user\n" + prior_intro + ' ' + prior_visual_text + "<|im_end|>\n"
            user_content = f"<|im_start|>user\n{question}<|im_end|>\n"
            assistant_content = "<|im_start|>assistant\n" if enable_thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            # NOTE: the prior study will always be in the front of the current study
            text.append(system_message + prior_visual_text + current_visual_text + user_content + assistant_content)
    
    visual_tokens = torch.cat(visual_tokens, dim=0)
    text = model.tokenizer(text, padding=True, return_tensors='pt', add_special_tokens=False)

    input_ids = text.input_ids
    attention_mask = text.attention_mask

    input_ids = input_ids.to(device=device)
    attention_mask = attention_mask.to(device=device)

    inputs_embeds = model.language_model.get_input_embeddings()(input_ids)
    image_pad_mask = (input_ids == 151655)  # NOTE: 151655 is "<|image_pad|>"
    inputs_embeds[image_pad_mask] = visual_tokens
    return inputs_embeds, attention_mask
    

@torch.inference_mode()
def generate(
    model, questions, 
    current_intros, current_scans, current_scan_types, current_num_scans, 
    prior_intros, prior_scans, prior_scan_types, prior_num_scans,
    encode_studies_independently, enable_thinking, **kwargs
):
    inputs_embeds, attention_mask = prepare_inputs_for_generation(
        model=model, batched_questions=questions, 
        batched_current_intro=current_intros,
        current_scans=current_scans, 
        current_scan_types=current_scan_types,
        current_num_scans=current_num_scans, 
        batched_prior_intro=prior_intros,
        prior_scans = prior_scans, 
        prior_scan_types=prior_scan_types,
        prior_num_scans=prior_num_scans,
        encode_studies_independently=encode_studies_independently,
        enable_thinking=enable_thinking
    )

    generated_ids = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **kwargs
    )
    
    return generated_ids
