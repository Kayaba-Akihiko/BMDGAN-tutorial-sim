#  Copyright (c) 2022. by Yi GU <gu.yi.gu4@is.naist.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.
import torch

from .protocol import DatasetProtocol
import numpy as np
from torch.utils.data import Dataset
from abc import ABC
from torchvision.datasets.utils import verify_str_arg, download_and_extract_archive
import os
import os.path
import pathlib
from PIL import Image
from utils import ConfigureHelper, ContainerHelper
from utils import ImageHelper
import numpy.typing as npt
from enum import Enum
from utils.typing import TypePathLike, TypeNPDTypeFloat, TypeNPDTypeUnsigned
from utils import MultiProcessingHelper
import skimage


class Stage(Enum):
    ONE = 1
    TWO = 2


class BaseDataset(Dataset, ABC, DatasetProtocol):
    """
    https://pytorch.org/vision/stable/_modules/torchvision/datasets/oxford_iiit_pet.html
    """

    _RESOURCES = (
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
         "5c4f3ee8e5d25df40f4fd59a7f44e54c"),
        ("https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
         "95a8c909bbe2e81eed6a22bccdf3f68f"),
    )

    NORMALIZATION_MEAN = np.array([0.485, 0.456, 0.406])
    NORMALIZATION_STD = np.array([0.229, 0.224, 0.225])

    LABEL_NAME_DICT = {0: "Foreground", 1: "Background", 2: "Not-classified"}

    def __init__(
            self,
            stage: Stage | str,  data_root: TypePathLike, split: str, ret_size: int | tuple[int, int],
            preload_dataset: bool, n_preload_worker=ConfigureHelper.max_n_worker, debug=False):
        super(BaseDataset, self).__init__()
        if isinstance(stage, str):
            self._stage = Stage[stage]
        else:
            self._stage = stage
        self._split = verify_str_arg(split, "split", ("trainval", "test"))
        self._base_folder = pathlib.Path(data_root) / "oxford-iiit-pet"
        self._images_folder = self._base_folder / "images"
        self._anns_folder = self._base_folder / "annotations"
        self._segs_folder = self._anns_folder / "trimaps"
        self._preload_dataset = preload_dataset
        self._preloaded = False
        self._image_load_func = self._load_image
        self._seg_load_func = self._load_seg
        self._ret_size = ContainerHelper.to_tuple(ret_size)

        if not self._check_exists():
            self._download()
        self._image_ids = self.read_image_ids(self._anns_folder / f"{self._split}.txt")
        self._image_id_idx_dict = {image_id: i for i, image_id in enumerate(self._image_ids)}

        self._images = [str(self._images_folder / f"{image_id}.jpg") for image_id in self._image_ids]
        self._segs = [str(self._segs_folder / f"{image_id}.png") for image_id in self._image_ids]

        if debug:
            self._images = self._images[:40]
            self._segs = self._segs[: 40]
            del self._image_id_idx_dict
            print("Running in debug mode.")

        if self._preload_dataset:
            mph = MultiProcessingHelper()
            self._images = mph.run(
                args=[(image,) for image in self._images], func=self._load_image, n_worker=n_preload_worker,
                desc=f"Preloading {split} images", mininterval=10, maxinterval=60)
            self._segs = mph.run(
                args=[(seg,) for seg in self._segs], func=self._load_seg, n_worker=n_preload_worker,
                desc=f"Preloading {split} labels", mininterval=10, maxinterval=60)
            self._image_load_func = self._identity
            self._seg_load_func = self._identity

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> dict[str, npt.NDArray[TypeNPDTypeFloat]]:
        image_id = self._image_ids[idx]
        image = self._image_load_func(self._images[idx])
        seg = self._seg_load_func(self._segs[idx])
        image, seg = self._augment(image=image, seg=seg)

        image = ImageHelper.resize(image, self._ret_size, order=1)  # (H, W, 3)
        seg = ImageHelper.resize(seg, self._ret_size, order=0)  # (H, W)

        if self._stage == Stage.ONE:
            mask = np.bitwise_or(seg == 1, seg == 3)
        else:
            assert self._stage == Stage.TWO, self._stage
            mask = seg == 1

        masked_image = image * mask[..., np.newaxis].astype(np.uint8)

        # add noise to image
        image = skimage.util.random_noise(image, mode="gaussian")
        image = image.clip(0., 1.)

        # normalization
        image = (image - self.NORMALIZATION_MEAN) / self.NORMALIZATION_STD
        masked_image = (masked_image - self.NORMALIZATION_MEAN) / self.NORMALIZATION_STD

        image = image.transpose((2, 0, 1)).astype(np.float32)
        masked_image = masked_image.transpose((2, 0, 1)).astype(np.float32)
        ret = {
            "image_id": image_id, "image": torch.from_numpy(image), "pet_image": torch.from_numpy(masked_image),
            "norm_mean": torch.from_numpy(self.NORMALIZATION_MEAN),
            "norm_std": torch.from_numpy(self.NORMALIZATION_STD)}
        return ret

    def get_item_by_image_id(self, image_id: str):
        idx = self._image_id_idx_dict[image_id]
        return self[idx]

    def _check_exists(self) -> bool:
        for folder in (self._images_folder, self._anns_folder):
            if not (os.path.exists(folder) and os.path.isdir(folder)):
                return False
        else:
            return True

    def _download(self) -> None:
        if self._check_exists():
            return
        for url, md5 in self._RESOURCES:
            download_and_extract_archive(url, download_root=str(self._base_folder), md5=md5)

    @classmethod
    def _load_image(cls, path) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB")).astype(float) / 255.

    @staticmethod
    def _load_seg(path) -> np.ndarray:
        return np.array(Image.open(path))  # 1, 2, 3 (pet, background, boarder)

    @staticmethod
    def read_image_ids(path) -> list[str]:
        image_ids = []
        with open(path) as file:
            for i, line in enumerate(file):
                image_id, *_ = line.strip().split()
                image_ids.append(image_id)
        return image_ids

    @staticmethod
    def _identity(x):
        return x

    def _augment(self, image: npt.NDArray[TypeNPDTypeFloat], seg: npt.NDArray[TypeNPDTypeUnsigned]) \
            -> tuple[npt.NDArray[TypeNPDTypeFloat], npt.NDArray[TypeNPDTypeUnsigned]]:
        return image, seg

