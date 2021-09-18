#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import platform
import typing as t
from pathlib import Path

CSIDL_PERSONAL: t.Final[int] = 5  # Мои документы
SHGFP_TYPE_CURRENT: t.Final[int] = 0  # Получить текущее значение
DATA_ROOT_OFFSET: t.Final[int] = 3


def get_data_root() -> Path:
    return Path(__file__).absolute().parents[DATA_ROOT_OFFSET] / "data"


def get_data_path(
    name: str,
    *args: str,
    root: Path = get_data_root(),
    create_path: bool = False,
    suffix: str = ".parquet",
) -> Path:
    names: t.List[str] = name.split(sep="::")
    for arg in args:
        names.extend(arg.split(sep="::"))

    path = root.joinpath(*names)
    if create_path:
        path.parent.mkdir(parents=True, exist_ok=True)

    if not path.suffix:
        path = path.with_suffix(suffix)

    return path


def get_output_root() -> Path:
    research_output = os.getenv("RESEARCH_OUTPUT")
    if research_output:
        output_path = Path(research_output)
        if not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        if output_path.is_dir():
            return output_path

    current_platform = platform.system().lower()
    if current_platform == "windows":
        import ctypes.wintypes

        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH)
        ctypes.windll.shell32.SHGetFolderPathW(
            None, CSIDL_PERSONAL, None, SHGFP_TYPE_CURRENT, buf
        )
        return Path(buf.value)

    elif current_platform == "linux" or current_platform == "darwin":
        return Path.home()

    else:
        raise RuntimeError(f"Unknown platform {current_platform}")
