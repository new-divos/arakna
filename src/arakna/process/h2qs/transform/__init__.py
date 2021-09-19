#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typing as t
from pathlib import Path

import click
from flask import current_app
from flask.cli import with_appcontext

from arakna.process.h2qs import h2qs
from arakna.process.path import get_data_root
from arakna.process.validation import validate_amount, validate_factor, validate_rate

__all__ = ["isotonic", "transform"]


@h2qs.group("transform")  # type: ignore
@click.option(
    "--rate",
    type=float,
    default=0.2,
    callback=validate_rate,
    help="The test sample rate",
)
@click.option(
    "--seed", type=int, default=43, help="The seed value for random generators"
)
@click.option(
    "--points_number",
    type=int,
    default=11,
    callback=validate_amount,
    help="The number of the control points",
)
@click.option(
    "--beta_plus",
    type=float,
    default=1.0,
    callback=validate_factor,
    help="The positive factor value for the penalty score",
)
@click.option(
    "--beta_minus",
    type=float,
    default=0.3,
    callback=validate_factor,
    help="The negative factor value for the penalty score",
)
@click.option(
    "--delta", type=int, default=20, help="The deadband width for the penalty score"
)
@click.option(
    "--processed_path",
    type=click.Path(exists=True),
    help="The path to the processed data",
)
@with_appcontext
def transform(
    rate: float,
    seed: int,
    points_number: int,
    beta_plus: float,
    beta_minus: float,
    delta: int,
    processed_path: t.Optional[str],
) -> None:
    current_app.config["TEST_RATE"] = rate
    current_app.logger.info(f"Using the test sample rate value {rate}")

    current_app.config["RANDOM_SEED"] = seed
    current_app.logger.info(f"Using a random generators seed value {seed}")

    current_app.config["POINTS_NUMBER"] = points_number
    current_app.config["BETA_PLUS"] = beta_plus
    current_app.config["BETA_MINUS"] = beta_minus
    current_app.config["DELTA"] = delta
    current_app.logger.info(
        "Using the penalty score parameters: "
        f"beta_plus = {beta_plus}, "
        f"beta_minus = {beta_minus}, "
        f"delta = {delta}"
    )

    processed_root = (
        Path(processed_path) if processed_path else get_data_root() / "processed"
    )
    current_app.config["PROCESSED_ROOT"] = processed_root


from arakna.process.h2qs.transform.isotonic import isotonic  # noqa: E402
