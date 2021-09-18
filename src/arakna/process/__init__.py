#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Blueprint

__all__ = ["h2qs", "path", "process"]

process = Blueprint("process", __name__)

from arakna.process.h2qs import h2qs  # noqa: E402
