#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import click


def validate_rate(_ctx: click.Context, param: str, value: float) -> float:
    if 0.0 < value < 1.0:
        return value
    else:
        raise click.BadParameter(f"the value of the {param} must be between 0 and 1")


def validate_factor(_ctx: click.Context, param: str, value: float) -> float:
    if value >= 0.0:
        return value
    else:
        raise click.BadParameter(f"the value of the {param} must be non-negative")


def validate_amount(_ctx: click.Context, param: str, value: int) -> int:
    # TODO: Задать нижнюю границу значения параметра
    if value > 0:
        return value
    else:
        raise click.BadParameter(f"the value of the {param} must be positive")
