#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from arakna.app import create_app
from arakna.config import get_config

app = create_app(get_config())
