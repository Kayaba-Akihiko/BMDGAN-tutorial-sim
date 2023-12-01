#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.

import torch
from network.hrnet import get_hrnet
from utils.typing import TypePathLike
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from dataset import Stage, TestDataset
from utils import ConfigureHelper
import pathlib
import tomllib
import argparse
import pandas as pd


class DataModule:

    def __init__(
            self, image_size,
            stage: Stage | str,  data_root: TypePathLike,
            test_batch_size: int | None = None,
            preload=True, n_worker=ConfigureHelper.max_n_worker, debug=False):
        self._n_worker = n_worker
        if isinstance(stage, str):
            self._stage = Stage[stage]
        else:
            self._stage = stage

        if test_batch_size is None:
            self._test_batch_size = 1
        else:
            self._test_batch_size = test_batch_size

        self._test_dataset = TestDataset(
            data_root=data_root, stage=stage, preload=preload, image_size=image_size, n_worker=self._n_worker, debug=debug)
        self.test_data_loader = DataLoader(
            dataset=self._test_dataset, batch_size=self._test_batch_size,
            shuffle=False, num_workers=self._n_worker, pin_memory=True)


class ModelTester:

    def __init__(
            self, data_module: DataModule, device: torch.device, output_dir: TypePathLike,
            pretrain_load_dir: TypePathLike, pretrain_load_prefix: str = 'ckp', model="hrnet48",):
        self._data_module = data_module
        self._output_dir = pathlib.Path(output_dir)
        self._device = device
        self._pretrain_load_dir = pathlib.Path(pretrain_load_dir)
        self._pretrain_load_prefix = pretrain_load_prefix

        self._nc = 3
        self._netG = get_hrnet(model, self._nc, self._nc).to(self._device)
        total_params = sum(p.numel() for p in self._netG.parameters())
        print(f"{self._netG.__class__} has {total_params * 1.e-6:.2f} M params.")

        self.load_model(self._pretrain_load_dir, self._pretrain_load_prefix)

        self.test_output_path = self._output_dir / "test_result.csv"

    @torch.no_grad()
    def test(self):

        result = []
        calc_psnr = PeakSignalNoiseRatio(reduction=None, dim=(1, 2, 3), data_range=1.)
        calc_ssim = StructuralSimilarityIndexMeasure(reduction=None, data_range=1.)
        iterator = self._data_module.test_data_loader
        n_iter = len(iterator)
        for data in tqdm(iterator, desc="Testing", mininterval=30, total=n_iter, maxinterval=60):
            batch_image_id = data["image_id"]
            batch_pet_image = data["pet_image"].to(self._device)
            batch_pred_pet_image = self._netG(data["image"].to(self._device))
            batch_norm_mean = data["norm_mean"][..., None, None].to(self._device)  # (B, C, 1, 1)
            batch_norm_std = data["norm_std"][..., None, None].to(self._device)  # (B, C, 1, 1)

            # denormalize -> (0, 1)
            batch_pet_image = (batch_pet_image * batch_norm_std + batch_norm_mean).clamp(0., 1.)
            batch_pred_pet_image = (batch_pred_pet_image * batch_norm_std + batch_norm_mean).clamp(0., 1.)

            batch_psnr = calc_psnr(batch_pred_pet_image, batch_pet_image)
            batch_ssim = calc_ssim(batch_pred_pet_image, batch_pet_image)

            n_batch = batch_pet_image.shape[0]
            for i in range(n_batch):
                image_id = batch_image_id[i]
                psnr = batch_psnr[i].item()
                ssim = batch_ssim[i].item()
                data = {"image_id": image_id, "psnr": psnr, "ssim": ssim}
                result.append(data)

        result = pd.DataFrame(result)
        result.sort_values(by='image_id', inplace=True)
        result.reset_index(drop=True, inplace=True)
        result.to_csv(self._output_dir / "test_result.csv")

    def load_model(self, load_dir: TypePathLike, prefix="ckp") -> None:
        load_dir = pathlib.Path(load_dir)
        load_path = load_dir / f"{prefix}_netG.pt"
        self._netG.load_state_dict(torch.load(load_path, map_location='cpu'))
        print(f"Model netG weights loaded from {load_path}.")


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
    output_dir.mkdir(exist_ok=True)

    image_size = config["image_size"]
    data_moudle_config = config["data_module_config"]
    tester_config = config["tester_config"]

    print("Configuring data module.")
    data_module = DataModule(image_size=image_size, n_worker=n_worker, **data_moudle_config)
    print("Configuring tester.")
    tester = ModelTester(output_dir=output_dir, data_module=data_module, device=device, **tester_config)

    print("Start testing.")
    tester.test()


if __name__ == '__main__':
    main()