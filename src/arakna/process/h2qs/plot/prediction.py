#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typing as t
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from flask import current_app
from flask.cli import with_appcontext

from arakna.process.h2qs.plot import plot
from arakna.process.path import get_image_name


@plot.command("prediction")  # type: ignore
@with_appcontext
def prediction() -> None:
    dataset = current_app.config["DATASET"]
    try:
        output_path = current_app.config["OUTPUT_ROOT"] / current_app.config["APP_NAME"]
        output_path.mkdir(parents=True, exist_ok=True)

        df = pd.read_parquet(current_app.config["DATASET_PATH"], engine="auto")
        if df.empty:
            raise RuntimeError("The dataset is empty")

        with open(current_app.config["CONFIG_FILE"], "r", encoding="utf-8") as fin:
            cfg = yaml.load(fin, Loader=yaml.CLoader)

        h2qs_cfg = cfg.get("h2qs", dict())
        config_name = h2qs_cfg.get("name", "[Unknown]")
        config_version = float(h2qs_cfg.get("version", 0.0))
        plot_cfg = h2qs_cfg.get("plot", dict())

        current_app.logger.info(f"Using plotting configuration '{config_name}'")
        if 1.0 <= config_version < 2.0:
            output_file = prediction_v1(
                df,
                plot_cfg,
                output_path,
                dataset,
                current_app.config["IMAGE_FORMAT"],
            )
            current_app.logger.info(f"The plot was saved to the file {output_file}")
        else:
            raise RuntimeError(
                f"Unknown plotting configuration version {config_version}"
            )

    except Exception as e:
        current_app.logger.error(
            f"The exception {e} occurs during plotting using the dataset {dataset}"
        )

    finally:
        current_app.logger.info("The plotting has been finished")


def prediction_v1(
    df: pd.DataFrame,
    plot_cfg: t.Dict[str, t.Any],
    output_path: Path,
    dataset: str,
    fmt: str,
) -> Path:
    width = plot_cfg.get("with", 10.0)
    height = plot_cfg.get("height", 8.0)

    title_cfg = plot_cfg.get("title", dict())
    xlabel_cfg = plot_cfg.get("xlabel", dict())
    ylabel_cfg = plot_cfg.get("ylabel", dict())
    legend_cfg = plot_cfg.get("legend", dict())
    grid_cfg = plot_cfg.get("grid", dict())
    prediction_cfg = plot_cfg.get("prediction", dict())

    fig, ax = plt.subplots(figsize=(width, height))

    qs_true_cfg = prediction_cfg.get("qs_true", dict())
    ax.scatter(df["hurst"], df["qs_true"], **qs_true_cfg)

    qs_pred_cfg = prediction_cfg.get("qs_pred", dict())
    ax.step(df["hurst"], df["qs_pred"], **qs_pred_cfg)

    cfg = prediction_cfg.get("title", dict())
    if cfg:
        text = cfg.get("text", None)
        if text is not None:
            del cfg["text"]
            ax.set_title(text, **title_cfg, **cfg)

    cfg = prediction_cfg.get("xlabel", dict())
    if cfg:
        text = cfg.get("text", None)
        if text is not None:
            del cfg["text"]
            ax.set_xlabel(text, **xlabel_cfg, **cfg)

    cfg = prediction_cfg.get("ylabel", dict())
    if cfg:
        text = cfg.get("text", None)
        if text is not None:
            del cfg["text"]
            ax.set_ylabel(text, **ylabel_cfg, **cfg)

    cfg = prediction_cfg.get("legend")
    if cfg:
        ax.legend(**legend_cfg, **cfg)

    ax.grid(True, **grid_cfg)

    output_file = output_path / get_image_name(dataset, fmt)
    fig.savefig(output_file, format=fmt)

    return output_file
