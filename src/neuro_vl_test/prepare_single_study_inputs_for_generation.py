import os
import sys
sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('..'))

import math
import torch


@torch.inference_mode()
def prepare_inputs_for_generation(model, batched_questions, scans, scan_types, num_scans, enable_thinking):
    device = scans.device

    # forward visual features for all studies in the batch
    features = model.forward_vision(scans, num_scans)
    features = model.forward_connector(features)
    features = features.to(dtype=torch.bfloat16)

    # prepare batch input
    text = []
    visual_tokens = []
    C = features.shape[-1]
    num_visual_tokens_per_scan = math.prod(model.visual_resolution)
    offset = torch.cumsum(num_scans, 0, dtype=torch.long) - num_scans
    for idx, questions_one_batch in enumerate(batched_questions): # questions_one_batch should be a list of questions (strings)
        visual_token = features[offset[idx]:offset[idx]+num_scans[idx].item()].reshape(-1, C)
        visual_tokens.append(visual_token[None, ...].expand(len(questions_one_batch), -1, -1).reshape(-1, C))

        for question in questions_one_batch:
            system_message = f"<|im_start|>system\n{model.system_message}<|im_end|>\n"
            visual_text = ''.join(
                (
                    f"Scan {i + 1} ({scan_types[offset[idx] + i]}): "
                    if scan_types[offset[idx] + i] != "unknown"
                    else f"Scan {i + 1}: "
                )
                + f"<|vision_start|>{'<|image_pad|>' * num_visual_tokens_per_scan}<|vision_end|>"
                for i in range(num_scans[idx].item())
            )
            user_content = f"<|im_start|>user\n{question}<|im_end|>\n"
            assistant_content = "<|im_start|>assistant\n" if enable_thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            text.append(system_message + "<|im_start|>user\n" + visual_text + "<|im_end|>\n" + user_content + assistant_content)
    
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
def generate(model, questions, scans, scan_types, num_scans, enable_thinking, **kwargs):
    inputs_embeds, attention_mask = prepare_inputs_for_generation(
        model=model, 
        batched_questions=questions, 
        scans=scans, 
        scan_types=scan_types,
        num_scans=num_scans, 
        enable_thinking=enable_thinking
    )

    generated_ids = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        **kwargs
    )
    
    return generated_ids
