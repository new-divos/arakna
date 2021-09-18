#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import typing as t

from flask import Flask

from arakna.config import ConfigProtocol

APP_NAME: t.Final[str] = "arakna"


def create_app(config: ConfigProtocol) -> Flask:
    app = Flask(APP_NAME)
    app.config.from_object(config)
    if hasattr(config, "init_app") and callable(config.init_app):
        config.init_app(app)

    from arakna.process import process as process_blueprint

    app.register_blueprint(process_blueprint)

    return app
