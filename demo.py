import os
import sys
sys.path.append(os.path.abspath('src'))

import argparse
import random
import re
import numpy as np
from pathlib import Path

import torch
from torchvision.transforms import Normalize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import SimpleITK as sitk
# NOTE: SimpleITK does not natively support data higher than 3 dimension.
# For a data with [x, y, z, t, c], 
# SimpleITK will focus on the first 3 dimension as spatial dimension,
# t (time) and c (channels) are typically flattened to a vector.
# As the result, GetArrayFromImage() returns a array of shape [z, y, x, t*c]

from neuro_vl.neuro_vl import VisionLanguageModel
from neuro_vl_test.prepare_single_study_inputs_for_generation import prepare_inputs_for_generation as prepare_single_study
from neuro_vl_test.prepare_paired_studies_inputs_for_generation import prepare_inputs_for_generation as prepare_paired_studies
from neuro_vl_test.utils import (
    hard_code_resize_crop,
    parse_output,
    set_model_bfloat16,
)


def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# arguments
def parse_args():
    parser = argparse.ArgumentParser('Neuro-VL demo', add_help=False)
    parser.add_argument('--model', default='/path/to/model/', type=str)
    parser.add_argument('--input', choices=['ReMIND-071/study_1', 'ReMIND-071/study_2', 'ReMIND-071', 'CQ500CT107'], required=True)
    parser.add_argument('--user_prompt', default='What do you see?', type=str)
    parser.add_argument('--provide_scan_type', action='store_true')
    return parser

# nifti helper
def reorient_to_rpi(img_sitk):
    """
    Reorientation from src -> RPI for the input img.

    Parameters:
        img_sitk: A SimpleITK image.

    Returns:
        img_rpi: The image reoriented to RPI.
        z_sitk:  Which axis (0/1/2) in img_rpi is the through-plane axis
                corresponding to the ORIGINAL index-z axis of img_sitk.
                (0 -> x, 1 -> y, 2 -> z)
    """
    # Physical direction of the ORIGINAL through-plane axis (original index-z)
    D0 = np.array(img_sitk.GetDirection(), dtype=float).reshape(3, 3)
    z0_phys = D0[:, 2]  # 3rd column = physical direction of original z-axis

    # Reorient to RPI
    orienter = sitk.DICOMOrientImageFilter()
    orienter.SetDesiredCoordinateOrientation('RPI')
    img_rpi = orienter.Execute(img_sitk)

    # Figure out where that original z-axis ended up in the RPI image
    D1 = np.array(img_rpi.GetDirection(), dtype=float).reshape(3, 3)
    dots = np.array([abs(np.dot(D1[:, j], z0_phys)) for j in range(3)])
    z_sitk = int(np.argmax(dots))  # 0/1/2 in the RPI image

    return img_rpi, z_sitk


def transpose_rpi_to_dhw(img_sitk, z_sitk):
    """ 
    Transpose img_sitk as:
    - D: Through-plane
    - H and W: In-plane
    Parameters:
        img_sitk: An axial-like sitk image of shape [x, y, z].
        z_sitk: through-plane axis.
    Return:
        img_arr: A numpy array after transposing axes.
        view: A string indicate the view.
    """
    img_arr = sitk.GetArrayFromImage(img_sitk) # [x, y, z] -> [z, y, x]
    d_arr = 2 - z_sitk
    
    if d_arr == 0: # z-axis has largest spacing -> axial (keep img_arr unchanged)
        view = 'axial'
    elif d_arr == 1: # y-axis has largest spacing -> coronal
        img_arr = np.transpose(img_arr, (1, 0, 2))  # (y, z, x)
        view = 'coronal'
    elif d_arr == 2: # x-axis has largest spacing -> sagittal
        img_arr = np.transpose(img_arr, (2, 0, 1))  # (x, z, y)
        view = 'sagittal'
    return img_arr, view


def hard_code_fix_dhw(img_arr, view):
    """
    Hard code the through-plane axis with the assumption that two in-plane sides should have the same size (Chenhui)
    Parameters:
        img_arr: An axial-like numpy array of shape [z, y, x].
        view: A string indicate the original view.
    Return:
        img_arr: A numpy array after transposing axes.
        view: A string indicate the view.
    """
    D, H, W = img_arr.shape

    if (H != D and W == D) or (H == D and W != D): # an abnormal thing happens in def::transpose_to_dhw
        if view == 'axial': # did not transpose in def::transpose_to_dhw
            if H != D and W == D:
                img_arr = np.transpose(img_arr, (1, 0, 2))
                view = 'coronal'
            if H == D and W != D:
                img_arr = np.transpose(img_arr, (2, 0, 1))
                view = 'sagittal'
        elif view == 'coronal': # transposed by (1, 0, 2) in def::transpose_to_dhw
            if H != D and W == D:
                img_arr = np.transpose(img_arr, (1, 0, 2))
                view = 'axial'
            if H == D and W != D:
                img_arr = np.transpose(img_arr, (2, 1, 0))
                view = 'sagittal'
        elif view == 'sagittal': # transposed by (2, 0, 1) in def::transpose_to_dhw
            if H != D and W == D:
                img_arr = np.transpose(img_arr, (1, 2, 0))
                view = 'axial'
            if H == D and W != D:
                img_arr = np.transpose(img_arr, (2, 1, 0))
                view = 'coronal'

    return img_arr, view


def clip_by_percentile(img, min, max):
    """ Percentile Clip
    """
    lower = np.percentile(img, min)
    upper = np.percentile(img, max)
    img = np.clip(img, lower, upper)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


def clip_by_window(img, level, width):
    """ Window clip.
    """
    lower = level - width / 2; upper = level + width / 2
    img = np.clip(img, lower, upper)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img


# load helper
def load_nifti(path, z_sitk=None):
    """
    Load a NIfTI file using SimpleITK, 
    Returns:
        (img, view)
    """

    # 1. Read via SimpleITK
    try:
        img_sitk = sitk.ReadImage(path)
    except Exception:
        return None
    
    # 2. Check it is 3D
    if img_sitk.GetDimension() != 3:
        return None
    if img_sitk.GetNumberOfComponentsPerPixel() != 1:
        return None

    # 3. Reorient to RPI 
    # NOTE: If z_sitk is {0, 1, 2},
    # this will not change z_sitk ONLY because we know
    # all NIfTI files have been reoriented 
    # in the order [L/R, P/A, S/I] (Asdur)
    if z_sitk == -1:
        img_sitk, z_sitk = reorient_to_rpi(img_sitk) # [x, y, z]
    else:
        img_sitk, _ = reorient_to_rpi(img_sitk)
    
    # 4. Convert SITK image to Numpy array
    if z_sitk == 0 or z_sitk == 1 or z_sitk == 2:
        img_arr, view = transpose_rpi_to_dhw(img_sitk, z_sitk)
    else:
        spacing_sitk = np.array(img_sitk.GetSpacing(), dtype=float)  # [x, y, z]
        z_sitk = np.argmax(spacing_sitk)
        img_arr, view = transpose_rpi_to_dhw(img_sitk, z_sitk)
        img_arr, view = hard_code_fix_dhw(img_arr, view)
    
    # 5. Filter out invalid scan
    if min(img_arr.shape) < 5:
        return None

    return img_arr, view


def load_study(scans, modality):
    normalizer = Normalize(torch.as_tensor(IMAGENET_DEFAULT_MEAN).mean(), torch.as_tensor(IMAGENET_DEFAULT_STD).mean())

    image = []
    for scan in scans:
        # NOTE: z_sitk is hard coded as None and will be inferred from spacing
        outputs = load_nifti(scan, z_sitk=None)
        if outputs is None:
            raise SystemExit(f'Failed to load NIfTI: {scan}. Check that the file is readable, contains a 3D scalar image, and has at least 5 voxels along each axis.')
        img_arr, _ = outputs

        if modality == 'mri':
            imgs = [clip_by_percentile(img_arr, 0.5, 99.5)]
        elif modality == 'ct':
            imgs = [
                clip_by_window(img_arr, 40, 80), # barin window
                clip_by_window(img_arr, 80, 200), # blood window
                clip_by_window(img_arr, 600, 2800) # bone window
            ]

        for img in imgs:
            img = torch.from_numpy(img * 255).to(torch.uint8)
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


def load_input(path, modality):
    path = Path(path)
    studies = [path] if any(p.is_file() for p in path.glob('*/img.nii.gz')) else sorted(
        p for p in path.iterdir() if p.is_dir()
    )
    if len(studies) not in (1, 2):
        raise ValueError('Expected one or two study folders.')

    scans = [
        sorted(str(p) for p in study.glob('*/img.nii.gz') if p.is_file())
        for study in studies
    ]
    if any(not study_scans for study_scans in scans):
        raise ValueError('Each study must contain {scan name}/img.nii.gz files.')

    if len(studies) == 2:
        prior_scans, current_scans = scans
        print('Studies are sorted alphabetically: first is prior, second is current.')
        print(f'Prior study: {studies[0]}')
    else:
        prior_scans = []
        current_scans = scans[0]
    print(f'Current study: {studies[-1]}')

    # NOTE: This demo assumes the prior and current studies share the same modality.
    # The model also supports studies with different modalities.
    prior_image = load_study(prior_scans, modality) if prior_scans else None
    current_image = load_study(current_scans, modality)

    prior_num_scans = len(prior_image) if prior_image is not None else None
    current_num_scans = len(current_image)

    return prior_image, prior_num_scans, current_image, current_num_scans


# metadata helper
# NOTE: The metadata below is hard coded for the included demo cases.
CQ500CT107_type = [
    'brain window CT', 'blood window CT', 'bone window CT',
    'brain window CT', 'blood window CT', 'bone window CT',
    'brain window CT', 'blood window CT', 'bone window CT',
    'brain window CT', 'blood window CT', 'bone window CT',
]
ReMIND_071_study_1_type = ['T2', 'T1 with contrast', 'T2']
ReMIND_071_study_2_type = ['T2', 'T1 with contrast']
ReMIND_071_study_1_intro = 'These are the images for Study 1, acquired in the same day before Study 2:'
ReMIND_071_study_2_intro = 'Study 2 was obtained after Study 1. These are its images:'
task_to_modality = {
    'ReMIND-071/study_1': 'mri',
    'ReMIND-071/study_2': 'mri',
    'ReMIND-071': 'mri',
    'CQ500CT107': 'ct',
}
task_to_type = {
    'ReMIND-071/study_1': ReMIND_071_study_1_type,
    'ReMIND-071/study_2': ReMIND_071_study_2_type,
    'ReMIND-071': ReMIND_071_study_2_type,
    'CQ500CT107': CQ500CT107_type,
}


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    random_seed(42)

    input_path = Path('./docs') / args.input
    modality = task_to_modality[args.input]
    prior_image, prior_num_scans, current_image, current_num_scans = load_input(input_path, modality)
    current_scan_types = task_to_type[args.input] if args.provide_scan_type else ['unknown'] * current_num_scans
    if prior_image is not None:
        prior_scan_types = ReMIND_071_study_1_type if args.provide_scan_type else ['unknown'] * prior_num_scans

    # create model
    model_dir = Path(args.model)
    model = VisionLanguageModel(
        system_message="You are Neuro-VL, created by MLiNS @ University of Michigan. You are a helpful assistant.",
        visual_resolution=(8, 7, 7),
        visual_encoder_name='vit_large_slice_scan_study_dualdinotxt1568',
        visual_encoder_path=str(model_dir / 'vit_large_slice_scan_study_dualdinotxt1568.pt'),
        connector_path=str(model_dir / 'connector_checkpoint.pt'),
        language_model_name_or_path=str(model_dir / 'Qwen3-8B'),
        freeze_visual_encoder=True,
        freeze_connector=True,
        freeze_language_model=True,
        gradient_checkpointing=False
    )
    random_seed(42)
    set_model_bfloat16(model)
    model.to(device)
    model.eval()

    # inference
    # NOTE: The input value below is hard coded for the included demo cases.
    with torch.inference_mode(), torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
        if prior_image is None:
            inputs_embeds, attention_mask = prepare_single_study(
                model=model,
                batched_questions=[[args.user_prompt]],
                scans=current_image.to(device=device, dtype=torch.bfloat16),
                scan_types=current_scan_types,
                num_scans=torch.tensor([current_num_scans], dtype=torch.long, device=device),
                enable_thinking=True,
            )
        else:
            inputs_embeds, attention_mask = prepare_paired_studies(
                model=model,
                batched_questions=[[args.user_prompt]],
                batched_current_intro=[ReMIND_071_study_2_intro],
                current_scans=current_image.to(device=device, dtype=torch.bfloat16),
                current_scan_types=current_scan_types,
                current_num_scans=torch.tensor([current_num_scans], dtype=torch.long, device=device),
                batched_prior_intro=[ReMIND_071_study_1_intro],
                prior_scans=prior_image.to(device=device, dtype=torch.bfloat16),
                prior_scan_types=prior_scan_types,
                prior_num_scans=torch.tensor([prior_num_scans], dtype=torch.long, device=device),
                encode_studies_independently=True,
                enable_thinking=True,
            )
            
        generated_ids = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_return_sequences=1,
            max_new_tokens=2048 if prior_image is None else 4096,
            do_sample=True,
            temperature=0.6,
            top_k=20,
            top_p=0.95,
            max_time=120.0,
        )

    generated_text = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    output = parse_output(generated_text)
    print(f"Thinking:\n{output['thinking']}")
    print(f"\nAnswer:\n{output['answer']}")
    

if __name__ == "__main__":
    main(parse_args().parse_args())
