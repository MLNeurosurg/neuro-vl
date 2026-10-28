import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_model_bfloat16(model):
    # https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/factory.py

    class LayerNormFp32(nn.LayerNorm):
        def forward(self, x: torch.Tensor):
            orig_type = x.dtype
            x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
            return x.to(orig_type)

    def _convert_ln(m):
        if isinstance(m, LayerNormFp32):
            m.weight.data = m.weight.data.to(torch.float32)
            m.bias.data = m.bias.data.to(torch.float32)

    model.to(dtype=torch.bfloat16)
    model.visual_encoder.apply(_convert_ln)


def parse_output(text):
    text = text.strip()
    match = re.match(r"^<think>\n(.*?)\n</think>\n\n(.*)$", text, re.DOTALL)
    if match is None:
        return {'thinking': '', 'answer': text}
    return {'thinking': match.group(1), 'answer': match.group(2)}


def set_data_device(batch, device):
    for k, v in batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device=device)
    return batch


def hard_code_resize_crop(img_tensor):
    """
    Resize to (48, H, W) with nearest-exact interpolation.
    Pad H/W to the longest side.
    Resize to (48, 256, 256), then center crop to (48, 224, 224).
    """
    if img_tensor.dtype != torch.float32:
        img_tensor = img_tensor.float()

    D, H, W = img_tensor.shape
    if D != 48:
        img_tensor = F.interpolate(
            img_tensor[None, None, ...],
            size=(48, H, W),
            mode='nearest-exact',
        ).squeeze(0).squeeze(0)

    if H != W:
        max_side = max(H, W)
        pad_h = max_side - H
        pad_w = max_side - W
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img_tensor = F.pad(img_tensor, (pad_left, pad_right, pad_top, pad_bottom), value=0)
        H = W = max_side

    img_tensor = F.interpolate(
        img_tensor[:, None, ...],
        size=(256, 256),
        mode='bilinear',
        align_corners=False,
    ).squeeze(1)

    return img_tensor[:, 16:240, 16:240]


def load_image(scans, normalizer):
    image = []
    for scan in scans:
        img = torch.load(scan, weights_only=True)
        img = img.float() / 255.0
        if tuple(img.shape) != (48, 224, 224):
            spatial_shape = img.shape[1:]
            nonzero_slice_mask = img.reshape(img.shape[0], -1).ne(0).any(dim=1)
            img = img[nonzero_slice_mask]
            if img.shape[0] == 0:
                img = torch.zeros((1, *spatial_shape), dtype=torch.float32)
            img = hard_code_resize_crop(img)
        img = normalizer(img[None, ...]).squeeze()
        image.append(img[None, ...])
    return torch.stack(image, dim=0).to(dtype=torch.float16)