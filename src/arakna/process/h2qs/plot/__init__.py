#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import typing as t
from pathlib import Path

import click
from flask import current_app
from flask.cli import with_appcontext

from arakna.process.h2qs import h2qs
from arakna.process.path import get_config_root

__all__ = ["plot", "prediction"]


@h2qs.group("plot")  # type: ignore
@click.argument("config", type=str, required=True)
@click.option(
    "-C", "--config_path", type=click.Path(exists=True), help="The config root path"
)
@click.option(
    "-m",
    "--image_format",
    type=click.Choice(["png", "svg", "jpeg"]),
    default="png",
    help="Image file format",
)
@with_appcontext
def plot(config: str, config_path: t.Optional[str], image_format: str) -> None:
    config_root = Path(config_path) if config_path else get_config_root()
    config_file = config_root / config

    if not config_file.exists():
        current_app.logger.error(f"The config {config_file} does not exist")
        sys.exit(1)
    elif not config_file.is_file():
        current_app.logger.error(f"The path {config_file} does not existlead to a file")
        sys.exit(1)

    current_app.config["CONFIG_FILE"] = config_file
    current_app.logger.info(f"Using the configuration file {config_file}")

    current_app.config["IMAGE_FORMAT"] = image_format
    current_app.logger.info(f"Using the image file format {image_format.upper()}")


from arakna.process.h2qs.plot.prediction import prediction  # noqa: E402
