#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import typing as t

from flask import Flask


class ConfigProtocol(t.Protocol):
    def init_app(self, app: Flask) -> None:
        ...


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", ":'(")

    def __init__(self):
        pass

    def init_app(self, _app: Flask) -> None:
        pass


def get_config() -> Config:
    return Config()
