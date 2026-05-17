# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for working with paths."""

import os
import shutil
from glob import glob
from itertools import islice
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import humanize

from vectorbtpro import _typing as tp

__all__ = [
    "list_any_files",
    "list_files",
    "list_dirs",
    "file_exists",
    "dir_exists",
    "file_size",
    "dir_size",
    "make_file",
    "make_dir",
    "remove_file",
    "remove_dir",
    "print_dir_tree",
]


def list_any_files(path: tp.Optional[tp.PathLike] = None, recursive: bool = False) -> tp.List[Path]:
    """List files and dirs matching a path.

    If the directory path is not provided, the current working directory is used."""
    if path is None:
        path = Path.cwd()
    else:
        path = Path(path)
    if path.exists() and path.is_dir():
        if recursive:
            path = path / "**" / "*"
        else:
            path = path / "*"
    return [Path(p) for p in glob(str(path), recursive=recursive)]


def list_files(path: tp.Optional[tp.PathLike] = None, recursive: bool = False) -> tp.List[Path]:
    """List files matching a path using `list_any_files`."""
    return [p for p in list_any_files(path, recursive=recursive) if p.is_file()]


def list_dirs(path: tp.Optional[tp.PathLike] = None, recursive: bool = False) -> tp.List[Path]:
    """List dirs matching a path using `list_any_files`."""
    return [p for p in list_any_files(path, recursive=recursive) if p.is_dir()]


def file_exists(file_path: tp.PathLike) -> bool:
    """Check whether a file exists."""
    file_path = Path(file_path)
    if file_path.exists() and file_path.is_file():
        return True
    return False


def dir_exists(dir_path: tp.PathLike) -> bool:
    """Check whether a directory exists."""
    dir_path = Path(dir_path)
    if dir_path.exists() and dir_path.is_dir():
        return True
    return False


def file_size(file_path: tp.PathLike, readable: bool = True, **kwargs) -> tp.Union[str, int]:
    """Get size of a file."""
    file_path = Path(file_path)
    if not file_exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found")
    n_bytes = file_path.stat().st_size
    if readable:
        return humanize.naturalsize(n_bytes, **kwargs)
    return n_bytes


def dir_size(dir_path: tp.PathLike, readable: bool = True, **kwargs) -> tp.Union[str, int]:
    """Get size of a directory."""
    dir_path = Path(dir_path)
    if not dir_exists(dir_path):
        raise FileNotFoundError(f"Directory '{dir_path}' not found")
    n_bytes = sum(path.stat().st_size for path in dir_path.glob("**/*") if path.is_file())
    if readable:
        return humanize.naturalsize(n_bytes, **kwargs)
    return n_bytes


def check_mkdir(
    dir_path: tp.PathLike,
    mkdir: tp.Optional[bool] = None,
    mode: tp.Optional[int] = None,
    parents: tp.Optional[bool] = None,
    exist_ok: tp.Optional[bool] = None,
) -> None:
    """Check whether the path to a directory exists and create if it doesn't.

    For defaults, see `mkdir` in `vectorbtpro._settings.path`."""
    from vectorbtpro._settings import settings

    mkdir_cfg = settings["path"]["mkdir"]

    if mkdir is None:
        mkdir = mkdir_cfg["mkdir"]
    if mode is None:
        mode = mkdir_cfg["mode"]
    if parents is None:
        parents = mkdir_cfg["parents"]
    if exist_ok is None:
        exist_ok = mkdir_cfg["exist_ok"]

    dir_path = Path(dir_path)
    if dir_path.exists() and not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    if not dir_path.exists() and not mkdir:
        raise FileNotFoundError(f"Directory '{dir_path}' not found. Use mkdir=True to proceed.")
    dir_path.mkdir(mode=mode, parents=parents, exist_ok=exist_ok)


def make_file(file_path: tp.PathLike, mode: int = 0o666, exist_ok: bool = True, **kwargs) -> Path:
    """Make an empty file."""
    file_path = Path(file_path)
    check_mkdir(file_path.parent, **kwargs)
    file_path.touch(mode=mode, exist_ok=exist_ok)
    return file_path


def make_dir(dir_path: tp.PathLike, **kwargs) -> Path:
    """Make an empty directory."""
    check_mkdir(dir_path, mkdir=True, **kwargs)
    return dir_path


def remove_file(file_path: tp.PathLike, missing_ok: bool = False) -> None:
    """Remove (delete) a file."""
    file_path = Path(file_path)
    if file_exists(file_path):
        file_path.unlink()
    elif not missing_ok:
        raise FileNotFoundError(f"File '{file_path}' not found")


def remove_dir(dir_path: tp.PathLike, missing_ok: bool = False, with_contents: bool = False) -> None:
    """Remove (delete) a directory."""
    dir_path = Path(dir_path)
    if dir_exists(dir_path):
        if any(dir_path.iterdir()) and not with_contents:
            raise ValueError(f"Directory '{dir_path}' has contents. Use with_contents=True to proceed.")
        shutil.rmtree(dir_path)
    elif not missing_ok:
        raise FileNotFoundError(f"Directory '{dir_path}' not found")


def get_common_prefix(paths: tp.Iterable[tp.PathLike]) -> str:
    """Returns the common prefix of a list of URLs or file paths."""
    if not paths:
        raise ValueError("The path list is empty")
    paths = [str(path) for path in paths]
    first = paths[0]
    parsed_first = urlparse(first)
    is_url = parsed_first.scheme != ""

    for path in paths:
        parsed = urlparse(path)
        if (parsed.scheme != parsed_first.scheme) or \
                (parsed.scheme != "" and parsed.netloc != parsed_first.netloc):
            return ""

    if is_url:
        parsed_urls = [urlparse(p) for p in paths]
        scheme = parsed_urls[0].scheme
        netloc = parsed_urls[0].netloc
        paths_split = [pu.path.strip("/").split("/") for pu in parsed_urls]
        min_length = min(len(p) for p in paths_split)
        common_components = []
        for i in range(min_length):
            current_component = paths_split[0][i]
            if all(p[i] == current_component for p in paths_split):
                common_components.append(current_component)
            else:
                break
        if common_components:
            common_path = "/" + "/".join(common_components) + "/"
        else:
            common_path = "/"
        common_url = urlunparse((scheme, netloc, common_path, "", "", ""))
        return common_url
    else:
        try:
            common_path = os.path.commonpath(paths)
            if not common_path.endswith(os.path.sep):
                common_path += os.path.sep
            return common_path
        except ValueError:
            return ""


def dir_tree_from_paths(
    paths: tp.Iterable[tp.PathLike],
    root: tp.Optional[tp.PathLike] = None,
    path_names: tp.Optional[tp.Iterable[str]] = None,
    root_name: tp.Optional[str] = None,
    level: int = -1,
    limit_to_dirs: bool = False,
    length_limit: tp.Optional[int] = 1000,
    sort: bool = True,
    space: str = "    ",
    branch: str = "│   ",
    tee: str = "├── ",
    last: str = "└── ",
) -> str:
    """Given paths, generate a visual tree structure."""
    resolved_paths = []
    for p in paths:
        if not isinstance(p, Path):
            parsed_url = urlparse(str(p))
            p = Path(parsed_url.path)
        resolved_paths.append(p.resolve())
    if path_names is None:
        path_names = [p.name for p in resolved_paths]
    path_display_map = {path: name for path, name in zip(resolved_paths, path_names)}
    if root is None:
        try:
            common_path_str = get_common_prefix(resolved_paths)
            root = Path(common_path_str).resolve()
        except ValueError:
            root = Path(".").resolve()
    else:
        if not isinstance(root, Path):
            parsed_url = urlparse(str(root))
            root = Path(parsed_url.path)
        root = root.resolve()

    dirs = set()
    path_set = set(resolved_paths)
    for path in resolved_paths:
        for parent in path.parents:
            if parent in path_set:
                dirs.add(parent)

    tree = {}
    for path in resolved_paths:
        try:
            relative_path = path.relative_to(root)
        except ValueError:
            continue
        if relative_path == Path("."):
            continue
        parts = relative_path.parts
        if not parts:
            continue
        current_level = tree
        for part in parts[:-1]:
            current_level = current_level.setdefault(part, {})
        last_part = parts[-1]
        if path in dirs:
            current_level.setdefault(last_part, {})
        else:
            current_level[last_part] = None

    files = 0
    dir_count = 0

    def _inner(current_tree, prefix="", current_lvl=-1, current_path=root):
        nonlocal files, dir_count
        if current_lvl == 0:
            return
        entries = list(current_tree.items())
        if sort:
            entries.sort(key=lambda x: (not isinstance(x[1], dict), x[0].lower()))
        if limit_to_dirs:
            entries = [e for e in entries if isinstance(e[1], dict)]
        pointers = [tee] * (len(entries) - 1) + [last] if entries else []
        for pointer, (name, subtree) in zip(pointers, entries):
            child_path = current_path / name
            display_name = path_display_map.get(child_path, name)
            yield prefix + pointer + display_name
            if isinstance(subtree, dict):
                dir_count += 1
                extension = branch if pointer == tee else space
                yield from _inner(
                    subtree,
                    prefix=prefix + extension,
                    current_lvl=(current_lvl - 1 if current_lvl > 0 else -1),
                    current_path=child_path,
                )
            elif not limit_to_dirs:
                files += 1

    tree_str = root_name if root_name is not None else root.name
    iterator = _inner(tree, current_lvl=level, current_path=root)
    if length_limit is not None:
        iterator = islice(iterator, length_limit)
    for line in iterator:
        tree_str += "\n" + line
    if next(iterator, None):
        tree_str += "\n" + f"... length_limit {length_limit} reached, counts:"
    tree_str += "\n" + f"\n{dir_count} directories" + (f", {files} files" if files else "")

    return tree_str


def dir_tree(dir_path: Path, **kwargs) -> str:
    """Generate a visual tree structure.

    Uses `dir_tree_from_paths`."""
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory '{dir_path}' not found")
    if not dir_path.is_dir():
        raise TypeError(f"Path '{dir_path}' is not a directory")
    paths = list(dir_path.rglob("*"))
    return dir_tree_from_paths(paths=paths, root=dir_path, **kwargs)


def print_dir_tree(*args, **kwargs) -> None:
    """Generate a directory tree with `tree` and print it out."""
    print(dir_tree(*args, **kwargs))
