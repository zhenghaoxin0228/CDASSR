# -*- coding: utf-8 -*-
# File: utils/file_ops.py
# Source: https://github.com/anson416/python-utilities

from pathlib import Path
from typing import Iterator, Optional


def is_dir(path: str) -> bool:
    return Path(path).is_dir()


def is_file(path: str) -> bool:
    return Path(path).is_file()


def get_file_ext(path: str) -> str:
    return Path(path).suffix


def iter_files(
    tgt_dir: str,
    exts: Optional[set[str]] = None,
    case_insensitive: bool = False,
    recursive: bool = False,
) -> Iterator[Path]:
    """
    Get all file paths under a directory (recursively). Similar to the `ls -a`
    (`ls -R` for recursive directory listing) command on Linux.

    Args:
        tgt_dir (str): Target directory.
        exts (Optional[Set[str]], optional): If not None, return a file only if
            its extension (including leading period) is in `exts`. Defaults to
            None.
        case_insensitive (bool, optional): Neglect case of file extensions.
            Used only if `exts` is not None. Defaults to False.
        recursive (bool, optional): Recurse into sub-directories. Defaults to
            False.

    Returns:
        Iterator[Path]: File paths under `tgt_dir`.
    """

    exts = set(map(lambda x: x.lower(), exts)) if exts is not None and case_insensitive else exts
    for child in (tgt_dir := Path(tgt_dir)).iterdir():
        if is_file(child):
            if exts is not None:
                ext = get_file_ext(child)
                if (ext.lower() if case_insensitive else ext) in exts:
                    yield child
            else:
                yield child
        elif is_dir(child):
            if recursive:
                yield from iter_files(child, exts=exts, case_insensitive=case_insensitive, recursive=recursive)
