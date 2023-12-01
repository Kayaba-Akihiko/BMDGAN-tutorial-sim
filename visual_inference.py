#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import torch
from network.hrnet import get_hrnet
from dataset import TestDataset
from utils import ConfigureHelper
import pathlib
import tomllib
import argparse
import pandas as pd
from einops import rearrange
import torch.nn.functional as F
from skimage.io import imsave
from tqdm import tqdm


def main():
    assert torch.cuda.is_available(), "GPU is not available."
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--n_worker", type=int, default=ConfigureHelper.max_n_worker)
    opt = parser.parse_args()

    config_path = pathlib.Path(opt.config_path)
    device = torch.device(opt.device)
    n_worker = opt.n_worker

    assert config_path.exists(), config_path
    with config_path.open(mode='rb') as f:
        config = tomllib.load(f)
    print(f"Config loaded from {config_path}.")
    print(config)

    output_dir = pathlib.Path("output") / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    visual_save_path = output_dir / "result.png"

    image_size = config["image_size"]
    dataset_config = config["dataset_config"]
    test_result_path = config["test_result_path"]
    pretrain_load_dir = pathlib.Path(config["pretrain_load_dir"])
    pretrain_load_prefix = config["pretrain_load_prefix"]
    model = config["model"]

    netG = get_hrnet(model, 3, 3).to(device)
    total_params = sum(p.numel() for p in netG.parameters())
    print(f"{netG.__class__} has {total_params * 1.e-6:.2f} M params.")
    load_path = pretrain_load_dir / f"{pretrain_load_prefix}_netG.pt"
    netG.load_state_dict(torch.load(load_path, map_location='cpu'))
    print(f"Model netG weights loaded from {load_path}.")

    test_dataset = TestDataset(image_size=image_size, n_worker=n_worker, **dataset_config)

    reference_metric = "psnr"
    asccending = False
    result_df = pd.read_csv(test_result_path, index_col=0)
    print(f"Test result loaded from {test_result_path}.")
    result_df.sort_values(by=reference_metric, ascending=asccending, inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    best_id = 0
    median_id = len(result_df.index) // 2
    worst_id = len(result_df.index) - 1

    result_df = result_df.iloc[[best_id, median_id, worst_id]]
    result_df.reset_index(drop=True, inplace=True)
    result_df["tag"] = ["best", "median", "wroest"]

    print(result_df)

    print("Start inference.")
    image_collection = []
    for _, row in tqdm(result_df.iterrows(), total=len(result_df.index), desc="Inferring"):
        image_id = row["image_id"]
        psnr = row["psnr"]
        ssim = row["ssim"]
        print(image_id, f"psnr:{psnr:.2f}", f"ssim:{ssim:.3f}")

        data = test_dataset.get_item_by_image_id(image_id)
        image = data["image"]  # (C, H, W)
        pet_image = data["pet_image"]  # (C, H, W)
        norm_mean = data["norm_mean"][..., None, None]  # (C, 1, 1)
        norm_std = data["norm_std"][..., None, None]  # (C, 1, 1)
        pred_pet_image = netG(image[None, ...].to(device))[0].cpu() # (C, H, W)

        # denormal  -> (0, 1) -> (0, 255.)
        image = ((image * norm_std + norm_mean) * 255.).clamp(0, 255.)
        pet_image = ((pet_image * norm_std + norm_mean) * 255.).clamp(0., 255.)
        pred_pet_image = ((pred_pet_image * norm_std + norm_mean) * 255.).clamp(0., 255.)
        ae_map = torch.abs(pred_pet_image - pet_image) # (0., 255.)

        image_collection += [image, pet_image, pred_pet_image, ae_map]

    image_collection = torch.stack(image_collection)  # (9, 3, H, W)
    pad_length = round(max(image_size) * 0.05)
    pads = (pad_length, pad_length, pad_length, pad_length, 0, 0, 0, 0)
    image_collection = F.pad(image_collection, pads, 'constant', 255)
    image_collection = rearrange(image_collection, 'b c h w -> b h w c')

    N, H, W, C = image_collection.shape
    n_col = 4
    n_row = N // n_col
    assert n_col * n_row == N
    result_image = torch.zeros((H * n_row, W * n_col, C), dtype=torch.uint8)
    for i in range(N):
        row_idx = i // n_col
        col_idx = i % n_col
        result_image[row_idx * H: (row_idx + 1) * H, col_idx * W: (col_idx + 1) * W] = image_collection[i]
    result_image = result_image.numpy()

    imsave(visual_save_path, result_image)
    print(f"Visualization result saved to {visual_save_path}.")


if __name__ == '__main__':
    main()