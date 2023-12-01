#  Copyright (c) 2023. by Yi GU <gu.yi.gu4@naist.ac.jp>, Imaging-based Computational Biomedicine Laboratory,
#  Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yi GU.


import pathlib
import tomllib
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():

    figure_size_px = (2048, 2048)
    dpi = 300

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config_path", type=str, required=True)
    opt = parser.parse_args()

    config_path = pathlib.Path(opt.config_path)

    assert config_path.exists(), config_path
    with config_path.open(mode='rb') as f:
        config = tomllib.load(f)
    print(f"Config loaded from {config_path}.")
    print(config)

    output_dir = pathlib.Path("output") / config_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    test_names = config['test_names']
    test_result_paths = config['test_result_paths']
    assert len(test_names) > 0, len(test_names)
    assert len(test_names) == len(test_result_paths), f"{len(test_names)}, {len(test_result_paths)}"

    result_df = []
    for test_name, test_result_path in zip(test_names, test_result_paths):
        test_result_df = pd.read_csv(test_result_path, index_col=0)
        test_result_df["test_name"] = test_name
        result_df.append(test_result_df)
    result_df = pd.concat(result_df, ignore_index=True)
    result_df.sort_values(by=["test_name", "image_id"], inplace=True)
    result_df.reset_index(drop=True, inplace=True)

    metrics = ['psnr', 'ssim']
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(figure_size_px[1] / dpi, figure_size_px[0] / dpi), dpi=dpi)
    for i, metric in enumerate(metrics):
        data = []
        for test_name in test_names:
            test_result_df = result_df[result_df["test_name"] == test_name]
            data.append(test_result_df[metric])
        axs[i].boxplot(data, labels=test_names, showmeans=True, showfliers=False)
        axs[i].set_ylabel(metric)
    fig.subplots_adjust(wspace=0.3)
    fig.savefig(output_dir / "result.png", dpi=dpi, bbox_inches="tight")


if __name__ == '__main__':
    main()