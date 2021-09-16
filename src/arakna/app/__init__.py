#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flask import Flask

from arakna.config import ConfigProtocol


def create_app(config: ConfigProtocol) -> Flask:
    app = Flask(__name__)
    app.config.from_object(config)
    if hasattr(config, "init_app") and callable(config.init_app):
        config.init_app(app)

    return app
