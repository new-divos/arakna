#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import typing as t
from pathlib import Path

from flask import Flask


class ConfigProtocol(t.Protocol):
    def __getitem__(self, key: str) -> t.Any:
        ...

    def init_app(self, app: Flask) -> None:
        ...


class Config:
    APP_NAME: t.Final[str] = "arakna"
    SECRET_KEY: t.Final[str] = os.getenv("SECRET_KEY", ":'(")
    LOGGING_PATH: t.Final[Path] = Path(
        os.getenv("LOGGING_PATH", str(Path.home().joinpath(".arakna", "global.log")))
    )

    def __init__(self):
        pass

    def __getitem__(self, key: str) -> t.Any:
        return self.__dict__.get(key) or self.__class__.__dict__.get(key)

    def init_app(self, _app: Flask) -> None:
        self.LOGGING_PATH.parent.mkdir(parents=True, exist_ok=True)


def get_config() -> Config:
    return Config()
