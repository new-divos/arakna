#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import sys
import typing as t
from pathlib import Path

import click
from flask import current_app
from flask.cli import with_appcontext

from arakna.process import process
from arakna.process.path import get_data_path, get_data_root, get_output_root

__all__ = ["h2qs", "plot", "transform"]


@process.cli.group("h2qs")  # type: ignore
@click.argument("dataset", type=str, required=True)
@click.option("--quiet", is_flag=True)
@click.option(
    "-D", "--data_path", type=click.Path(exists=True), help="The data root path"
)
@click.option(
    "-O", "--output_path", type=click.Path(exists=True), help="The output root path"
)
@with_appcontext
def h2qs(
    dataset: str, quiet: bool, data_path: t.Optional[str], output_path: t.Optional[str]
) -> None:
    current_app.logger.setLevel(logging.ERROR if quiet else logging.INFO)
    data_root = Path(data_path) if data_path else get_data_root()
    dataset_path = get_data_path(dataset, root=data_root)

    if not dataset_path.exists():
        current_app.logger.error(
            f"The dataset {dataset} ({dataset_path}) does not exist"
        )
        sys.exit(1)
    elif not dataset_path.is_file():
        current_app.logger.error(f"The path {dataset_path} does not lead to a file")
        sys.exit(1)

    output_root = Path(output_path) if output_path else get_output_root()

    current_app.config["DATA_ROOT"] = data_root
    current_app.config["DATASET"] = dataset
    current_app.config["DATASET_PATH"] = dataset_path
    current_app.config["OUTPUT_ROOT"] = output_root

    current_app.logger.info(f"Using the dataset {dataset} from the file {dataset_path}")


from arakna.process.h2qs.plot import plot  # noqa: E402
from arakna.process.h2qs.transform import transform  # noqa: E402
