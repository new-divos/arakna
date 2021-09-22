#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging

from flask import Flask

from arakna.config import ConfigProtocol


def create_app(config: ConfigProtocol) -> Flask:
    log_format_str = (
        "[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} "
        "%(levelname)s - %(message)s"
    )
    logging.basicConfig(
        level=logging.INFO,
        format=log_format_str,
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=config["LOGGING_PATH"],
    )

    app = Flask(config["APP_NAME"])

    app.config.from_object(config)
    if hasattr(config, "init_app") and callable(config.init_app):
        config.init_app(app)

    from arakna.process import process as process_blueprint

    app.register_blueprint(process_blueprint)

    return app
