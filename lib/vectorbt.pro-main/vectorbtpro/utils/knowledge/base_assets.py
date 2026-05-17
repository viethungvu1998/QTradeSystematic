# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Base asset classes."""

import re
import json
from collections.abc import MutableSequence
from pathlib import Path

import pandas as pd

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks, search
from vectorbtpro.utils.config import Configured
from vectorbtpro.utils.pickling import decompress, load_bytes
from vectorbtpro.utils.path_ import check_mkdir, remove_dir, dir_tree_from_paths
from vectorbtpro.utils.pbar import ProgressBar
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc, Sub
from vectorbtpro.utils.config import flat_merge_dicts, deep_merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.execution import Task, execute, NoResult
from vectorbtpro.utils.module_ import get_caller_qualname
from vectorbtpro.utils.decorators import hybrid_method

try:
    if not tp.TYPE_CHECKING:
        raise ImportError
    from tiktoken import Encoding as EncodingT
except ImportError:
    EncodingT = tp.Any

__all__ = [
    "KnowledgeAsset",
]


# ############# Asset classes ############# #


KnowledgeAssetT = tp.TypeVar("KnowledgeAssetT", bound="KnowledgeAsset")


class KnowledgeAsset(Configured, MutableSequence):
    """Class for working with a knowledge asset.

    This class behaves like a mutable sequence.

    For defaults, see `vectorbtpro._settings.knowledge`."""

    _settings_path: tp.SettingsPath = "knowledge"

    _expected_keys: tp.ExpectedKeys = (Configured._expected_keys or set()) | {
        "data",
        "single_item",
    }

    @classmethod
    def stack(
        cls: tp.Type[KnowledgeAssetT],
        *objs: tp.MaybeTuple[KnowledgeAssetT],
        **kwargs,
    ) -> KnowledgeAssetT:
        """Stack multiple `KnowledgeAsset` instances."""
        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, KnowledgeAsset):
                raise TypeError("Each object to be stacked must be an instance of KnowledgeAsset")
        new_data = []
        new_single_item = True
        for obj in objs:
            new_data.extend(obj.data)
            if not obj.single_item:
                new_single_item = False
        return cls(data=new_data, single_item=new_single_item, **kwargs)

    @hybrid_method
    def merge(
        cls_or_self: tp.MaybeType[KnowledgeAssetT],
        *objs: tp.MaybeTuple[KnowledgeAssetT],
        flatten_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Either merge multiple `KnowledgeAsset` instances if called as a class method or instance
        method with at least one additional object, or merge data items of a single instance if called
        as an instance method with no additional objects.

        Usage:
            ```pycon
            >>> asset1 = asset.select(["s"])
            >>> asset2 = asset.select(["b", "d2"])
            >>> asset1.merge(asset2).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]
            ```
        """
        if not isinstance(cls_or_self, type) and len(objs) == 0:
            if isinstance(cls_or_self[0], list):
                return cls_or_self.merge_lists(**kwargs)
            if isinstance(cls_or_self[0], dict):
                return cls_or_self.merge_dicts(**kwargs)
            raise ValueError("Cannot determine type of data items. Use merge_lists or merge_dicts.")
        elif not isinstance(cls_or_self, type) and len(objs) > 0:
            objs = (cls_or_self, *objs)
            cls = type(cls_or_self)
        else:
            cls = cls_or_self

        if len(objs) == 1:
            objs = objs[0]
        objs = list(objs)
        for obj in objs:
            if not checks.is_instance_of(obj, KnowledgeAsset):
                raise TypeError("Each object to be merged must be an instance of KnowledgeAsset")

        if flatten_kwargs is None:
            flatten_kwargs = {}
        if "annotate_all" not in flatten_kwargs:
            flatten_kwargs["annotate_all"] = True
        if "excl_types" not in flatten_kwargs:
            flatten_kwargs["excl_types"] = (tuple, set, frozenset)
        max_items = 1
        new_single_item = True
        for obj in objs:
            obj_data = obj.data
            if len(obj_data) > max_items:
                max_items = len(obj_data)
            if not obj.single_item:
                new_single_item = False
        flat_data = []
        for obj in objs:
            obj_data = obj.data
            if len(obj_data) == 1:
                obj_data = [obj_data] * max_items
            flat_obj_data = list(map(lambda x: search.flatten_obj(x, **flatten_kwargs), obj_data))
            flat_data.append(flat_obj_data)
        new_data = []
        for flat_dcts in zip(*flat_data):
            merged_flat_dct = flat_merge_dicts(*flat_dcts)
            new_data.append(search.unflatten_obj(merged_flat_dct))
        return cls(data=new_data, single_item=new_single_item, **kwargs)

    @classmethod
    def from_json_file(
        cls: tp.Type[KnowledgeAssetT],
        path: tp.PathLike,
        compression: tp.Union[None, bool, str] = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Build `KnowledgeAsset` from a JSON file."""
        bytes_ = load_bytes(path, compression=compression, decompress_kwargs=decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(data=json.loads(json_str), **kwargs)

    @classmethod
    def from_json_bytes(
        cls: tp.Type[KnowledgeAssetT],
        bytes_: bytes,
        compression: tp.Union[None, bool, str] = None,
        decompress_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> KnowledgeAssetT:
        """Build `KnowledgeAsset` from JSON bytes."""
        if decompress_kwargs is None:
            decompress_kwargs = {}
        bytes_ = decompress(bytes_, compression=compression, **decompress_kwargs)
        json_str = bytes_.decode("utf-8")
        return cls(data=json.loads(json_str), **kwargs)

    def __init__(self, data: tp.List[tp.Any], single_item: bool = True, **kwargs) -> None:
        if not isinstance(data, list):
            data = [data]
        else:
            data = list(data)
        if len(data) > 1:
            single_item = False

        Configured.__init__(
            self,
            data=data,
            single_item=single_item,
            **kwargs,
        )

        self._data = data
        self._single_item = single_item

    @property
    def data(self) -> tp.List[tp.Any]:
        """Data."""
        return self._data

    @property
    def single_item(self) -> bool:
        """Whether this instance holds a single item."""
        return self._single_item

    def modify_data(self, data: tp.List[tp.Any]) -> None:
        """Modify data in place."""
        if len(data) > 1:
            single_item = False
        else:
            single_item = self.single_item
        self._data = data
        self._single_item = single_item
        self.update_config(data=data, single_item=single_item)

    # ############# Item methods ############# #

    def get_item(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]]) -> tp.Any:
        """Get a data item or a selection of data items."""
        if checks.is_complex_iterable(index):
            if all(checks.is_bool(i) for i in index):
                index = list(index)
                if len(index) != len(self.data):
                    raise IndexError("Boolean index must have the same length as data")
                return self.replace(data=[item for item, flag in zip(self.data, index) if flag])
            if all(checks.is_int(i) for i in index):
                return self.replace(data=[self.data[i] for i in index])
            raise TypeError("Index must contain all integers or all booleans")
        if isinstance(index, slice):
            return self.replace(data=self.data[index])
        return self.data[index]

    def set_item(
        self: KnowledgeAssetT,
        index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]],
        value: tp.Any,
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Set a data item or a selection of data items.

        Returns a new `KnowledgeAsset` instance if `inplace` is False."""
        new_data = list(self.data)
        if checks.is_complex_iterable(index):
            index = list(index)
            if all(checks.is_bool(i) for i in index):
                if len(index) != len(new_data):
                    raise IndexError("Boolean index must have the same length as data")
                if checks.is_complex_iterable(value):
                    value = list(value)
                    if len(value) == len(index):
                        for i, (b, v) in enumerate(zip(index, value)):
                            if b:
                                new_data[i] = v
                    else:
                        num_true = sum(index)
                        if len(value) != num_true:
                            raise ValueError(f"Attempting to assign {len(value)} values to {num_true} targets")
                        it = iter(value)
                        for i, b in enumerate(index):
                            if b:
                                new_data[i] = next(it)
                else:
                    for i, b in enumerate(index):
                        if b:
                            new_data[i] = value
            elif all(checks.is_int(i) for i in index):
                if checks.is_complex_iterable(value):
                    value = list(value)
                    if len(value) != len(index):
                        raise ValueError(f"Attempting to assign {len(value)} values to {len(index)} targets")
                    for i, v in zip(index, value):
                        new_data[i] = v
                else:
                    for i in index:
                        new_data[i] = value
            else:
                raise TypeError("Index must contain all integers or all booleans")
        else:
            new_data[index] = value
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def remove_item(
        self: KnowledgeAssetT,
        index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]],
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Remove a data item or a selection of data items.

        Returns a new `KnowledgeAsset` instance if `inplace` is False."""
        new_data = list(self.data)
        if checks.is_complex_iterable(index):
            if all(checks.is_bool(i) for i in index):
                index = list(index)
                if len(index) != len(new_data):
                    raise IndexError("Boolean index must have the same length as data")
                new_data = [item for item, flag in zip(new_data, index) if not flag]
            elif all(checks.is_int(i) for i in index):
                indices_to_remove = set(index)
                max_index = len(new_data) - 1
                for i in indices_to_remove:
                    if not -len(new_data) <= i <= max_index:
                        raise IndexError(f"Index {i} out of range")
                new_data = [item for i, item in enumerate(new_data) if i not in indices_to_remove]
            else:
                raise TypeError("Index must contain all integers or all booleans")
        else:
            del new_data[index]
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def append_item(
        self: KnowledgeAssetT,
        d: tp.Any,
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Append a data item.

        Returns a new `KnowledgeAsset` instance if `inplace` is False."""
        new_data = list(self.data)
        new_data.append(d)
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def extend_items(
        self: KnowledgeAssetT,
        data: tp.Iterable[tp.Any],
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Append a data item.

        Returns a new `KnowledgeAsset` instance if `inplace` is False."""
        new_data = list(self.data)
        new_data.extend(data)
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def remove_empty(self, inplace: bool = False) -> tp.Optional[KnowledgeAssetT]:
        """Remove empty data items."""
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc

        new_data = [d for d in self.data if not FindRemoveAssetFunc.is_empty_func(d)]
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def unique(
        self: KnowledgeAssetT,
        *args,
        keep: str = "first",
        inplace: bool = False,
        **kwargs,
    ) -> tp.Optional[KnowledgeAssetT]:
        """De-duplicate based on `KnowledgeAsset.get` called on `*args` and `**kwargs`.

        Returns a new `KnowledgeAsset` instance if `inplace` is False.

        Usage:
            ```pycon
            >>> asset.unique("b").get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]
            ```
        """
        keys = self.get(*args, **kwargs)
        if keep.lower() == "first":
            seen = set()
            new_data = []
            for key, item in zip(keys, self.data):
                if key not in seen:
                    seen.add(key)
                    new_data.append(item)
        elif keep.lower() == "last":
            seen = set()
            new_data_reversed = []
            for key, item in zip(reversed(keys), reversed(self.data)):
                if key not in seen:
                    seen.add(key)
                    new_data_reversed.append(item)
            new_data = list(reversed(new_data_reversed))
        else:
            raise ValueError(f"Invalid keep option: '{keep}'")
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def sort(
        self: KnowledgeAssetT,
        *args,
        keys: tp.Optional[tp.Iterable[tp.Key]] = None,
        ascending: bool = True,
        inplace: bool = False,
        **kwargs,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Sort based on `KnowledgeAsset.get` called on `*args` and `**kwargs`.

        Returns a new `KnowledgeAsset` instance if `inplace` is False.

        Usage:
            ```pycon
            >>> asset.sort("d2.c").get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]
            ```
        """
        if keys is None:
            keys = self.get(*args, **kwargs)
        new_data = [x for _, x in sorted(zip(keys, self.data), key=lambda x: x[0], reverse=not ascending)]
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def shuffle(
        self: KnowledgeAssetT,
        seed: tp.Optional[int] = None,
        inplace: bool = False,
    ) -> tp.Optional[KnowledgeAssetT]:
        """Shuffle data items."""
        import random

        if seed is not None:
            random.seed(seed)
        new_data = list(self.data)
        random.shuffle(new_data)
        if inplace:
            self.modify_data(new_data)
            return None
        return self.replace(data=new_data)

    def sample(
        self,
        k: tp.Optional[int] = None,
        seed: tp.Optional[int] = None,
        wrap: bool = True,
    ) -> tp.Any:
        """Pick a random sample of data items."""
        import random

        if k is None:
            k = 1
            single_item = True
        else:
            single_item = False
        if seed is not None:
            random.seed(seed)
        new_data = random.sample(self.data, min(len(self.data), k))
        if wrap:
            return self.replace(data=new_data, single_item=single_item)
        if single_item:
            return new_data[0]
        return new_data

    def print_sample(self, k: tp.Optional[int] = None, seed: tp.Optional[int] = None, **kwargs) -> None:
        """Print a random sample.

        Keyword arguments are passed to `KnowledgeAsset.print`."""
        self.sample(k=k, seed=seed).print(**kwargs)

    # ############# Collection methods ############# #

    def __len__(self) -> int:
        return len(self.data)

    # ############# Sequence methods ############# #

    def __getitem__(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]]) -> tp.Any:
        return self.get_item(index)

    # ############# MutableSequence methods ############# #

    def insert(self, index: int, value: tp.Any) -> None:
        new_data = list(self.data)
        new_data.insert(index, value)
        self.modify_data(new_data)

    def __setitem__(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]], value: tp.Any) -> None:
        self.set_item(index, value, inplace=True)

    def __delitem__(self, index: tp.Union[int, slice, tp.Iterable[tp.Union[bool, int]]]) -> None:
        self.remove_item(index, inplace=True)

    def __add__(self: KnowledgeAssetT, other: tp.Any) -> KnowledgeAssetT:
        if not isinstance(other, KnowledgeAsset):
            other = KnowledgeAsset(other)
        mro_self = self.__class__.mro()
        mro_other = other.__class__.mro()
        common_bases = set(mro_self).intersection(mro_other)
        for cls in mro_self:
            if cls in common_bases:
                new_type = cls
                break
        else:
            new_type = KnowledgeAsset
        return new_type.stack(self, other)

    def __iadd__(self: KnowledgeAssetT, other: tp.Any) -> KnowledgeAssetT:
        if isinstance(other, KnowledgeAsset):
            other = other.data
        self.extend_items(other, inplace=True)
        return self

    # ############# Apply methods ############# #

    def apply(
        self,
        func: tp.MaybeList[tp.Union[tp.AssetFuncLike, tp.AssetPipeline]],
        *args,
        execute_kwargs: tp.KwargsLike = None,
        wrap: tp.Optional[bool] = None,
        single_item: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Apply a function to each data item.

        Function can be either a callable, a tuple of function and its arguments,
        a `vectorbtpro.utils.execution.Task` instance, a subclass of
        `vectorbtpro.utils.knowledge.base_asset_funcs.AssetFunc` or its prefix or full name.
        Moreover, function can be a list of the above. In such a case, `BasicAssetPipeline` will be used.
        If function is a valid expression, `ComplexAssetPipeline` will be used.

        Uses `vectorbtpro.utils.execution.execute` for execution.

        If `wrap` is True, returns a new `KnowledgeAsset` instance, otherwise raw output.

        Usage:
            ```pycon
            >>> asset.apply(["flatten", ("query", len)])
            [5, 5, 5, 5, 6]

            >>> asset.apply("query(flatten(d), len)")
            [5, 5, 5, 5, 6]
            ```
        """
        from vectorbtpro.utils.knowledge.asset_pipelines import AssetPipeline, BasicAssetPipeline, ComplexAssetPipeline

        execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)
        asset_func_meta = {}

        if isinstance(func, list):
            func, args, kwargs = (
                BasicAssetPipeline(
                    func,
                    *args,
                    cond_kwargs=dict(asset=self),
                    asset_func_meta=asset_func_meta,
                    **kwargs,
                ),
                (),
                {},
            )
        elif isinstance(func, str) and not func.isidentifier():
            if len(args) > 0:
                raise ValueError("No more positional arguments can be applied to ComplexAssetPipeline")
            func, args, kwargs = (
                ComplexAssetPipeline(
                    func,
                    context=kwargs.get("template_context", None),
                    cond_kwargs=dict(asset=self),
                    asset_func_meta=asset_func_meta,
                    **kwargs,
                ),
                (),
                {},
            )
        elif not isinstance(func, AssetPipeline):
            func, args, kwargs = AssetPipeline.resolve_task(
                func,
                *args,
                cond_kwargs=dict(asset=self),
                asset_func_meta=asset_func_meta,
                **kwargs,
            )
        else:
            if len(args) > 0:
                raise ValueError("No more positional arguments can be applied to AssetPipeline")
            if len(kwargs) > 0:
                raise ValueError("No more keyword arguments can be applied to AssetPipeline")
        prefix = get_caller_qualname().split(".")[-1]
        if "_short_name" in asset_func_meta:
            prefix += f"[{asset_func_meta['_short_name']}]"
        elif isinstance(func, type):
            prefix += f"[{func.__name__}]"
        else:
            prefix += f"[{type(func).__name__}]"
        execute_kwargs = deep_merge_dicts(
            dict(
                show_progress=False if self.single_item else None,
                pbar_kwargs=dict(
                    bar_id=get_caller_qualname(),
                    prefix=prefix,
                ),
            ),
            execute_kwargs,
        )

        def _get_task_generator():
            for i, d in enumerate(self.data):
                _kwargs = dict(kwargs)
                if "template_context" in _kwargs:
                    _kwargs["template_context"] = flat_merge_dicts(
                        {"i": i},
                        _kwargs["template_context"],
                    )
                yield Task(func, d, *args, **_kwargs)

        tasks = _get_task_generator()
        new_data = execute(tasks, size=len(self.data), **execute_kwargs)
        if new_data is NoResult:
            new_data = []
        if wrap is None and asset_func_meta.get("_wrap", None) is not None:
            wrap = asset_func_meta["_wrap"]
        if wrap is None:
            wrap = True
        if single_item is None:
            single_item = self.single_item
        if wrap:
            return self.replace(data=new_data, single_item=single_item)
        if single_item:
            if len(new_data) == 1:
                return new_data[0]
            if len(new_data) == 0:
                return None
        return new_data

    def get(
        self: KnowledgeAssetT,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Get data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.GetAssetFunc`.

        Use argument `path` to specify what part of the data item should be got. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, the data item will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the data item, will skip the data item.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the index of the data item is represented by "i", the data item itself is represented by "d",
        the data item under the path is represented by "x" while its fields are represented by their names.

        Usage:
            ```pycon
            >>> asset.get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.get("d2.l[0]")
            [1, 3, 5, 7, 9]

            >>> asset.get("d2.l", source=lambda x: sum(x))
            [3, 7, 11, 15, 19]

            >>> asset.get("d2.l[0]", keep_path=True)
            [{'d2': {'l': {0: 1}}},
             {'d2': {'l': {0: 3}}},
             {'d2': {'l': {0: 5}}},
             {'d2': {'l': {0: 7}}},
             {'d2': {'l': {0: 9}}}]

            >>> asset.get(["d2.l[0]", "d2.l[1]"])
            [{'d2': {'l': {0: 1, 1: 2}}},
             {'d2': {'l': {0: 3, 1: 4}}},
             {'d2': {'l': {0: 5, 1: 6}}},
             {'d2': {'l': {0: 7, 1: 8}}},
             {'d2': {'l': {0: 9, 1: 10}}}]

            >>> asset.get("xyz", skip_missing=True)
            [123]
            ```
        """
        if path is None and source is None:
            if self.single_item:
                return self.data[0]
            return self.data
        return self.apply(
            "get",
            path=path,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            template_context=template_context,
            **kwargs,
        )

    def select(self: KnowledgeAssetT, *args, **kwargs) -> KnowledgeAssetT:
        """Call `KnowledgeAsset.get` and return a new `KnowledgeAsset` instance."""
        return self.get(*args, wrap=True, **kwargs)

    def set(
        self: KnowledgeAssetT,
        value: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Set data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.SetAssetFunc`.

        Argument `value` can be any value, function (will become a template), or a template. In this template,
        the index of the data item is represented by "i", the data item itself is represented by "d",
        the data item under the path is represented by "x" while its fields are represented by their names.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to template substitution in `value`.

        Usage:
            ```pycon
            >>> asset.set(lambda d: sum(d["d2"]["l"])).get()
            [3, 7, 11, 15, 19]

            >>> asset.set(lambda d: sum(d["d2"]["l"]), path="d2.sum").get()
            >>> asset.set(lambda x: sum(x["l"]), path="d2.sum").get()
            >>> asset.set(lambda l: sum(l), path="d2.sum").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2], 'sum': 3}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4], 'sum': 7}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6], 'sum': 11}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8], 'sum': 15}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10], 'sum': 19}, 'xyz': 123}]

            >>> asset.set(lambda l: sum(l), path="d2.l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': 3}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': 7}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': 11}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': 15}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': 19}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "set",
            value=value,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            template_context=template_context,
            **kwargs,
        )

    def remove(
        self: KnowledgeAssetT,
        path: tp.MaybeList[tp.PathLikeKey],
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Remove data items or parts of them.

        If `path` is an integer, removes the entire data item at that index.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.RemoveAssetFunc`.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Usage:
            ```pycon
            >>> asset.remove("d2.l[0]").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [10]}, 'xyz': 123}]

            >>> asset.remove("xyz", skip_missing=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]
            ```
        """
        return self.apply(
            "remove",
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def move(
        self: KnowledgeAssetT,
        path: tp.Union[tp.PathMoveDict, tp.MaybeList[tp.PathLikeKey]],
        new_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Move data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.MoveAssetFunc`.

        Use argument `path` to specify what part of the data item should be renamed. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Use argument `new_path` to specify the last part of the data item (i.e., token) that should be renamed to.
        Multiple tokens can be provided. If None, `path` must be a dictionary.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Usage:
            ```pycon
            >>> asset.move("d2.l", "l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red'}, 'l': [1, 2]},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue'}, 'l': [3, 4]},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green'}, 'l': [5, 6]},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow'}, 'l': [7, 8]},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black'}, 'xyz': 123, 'l': [9, 10]}]

            >>> asset.move({"d2.c": "c", "b": "d2.b"}).get()
            >>> asset.move(["d2.c", "b"], ["c", "d2.b"]).get()
            [{'s': 'ABC', 'd2': {'l': [1, 2], 'b': True}, 'c': 'red'},
             {'s': 'BCD', 'd2': {'l': [3, 4], 'b': True}, 'c': 'blue'},
             {'s': 'CDE', 'd2': {'l': [5, 6], 'b': False}, 'c': 'green'},
             {'s': 'DEF', 'd2': {'l': [7, 8], 'b': False}, 'c': 'yellow'},
             {'s': 'EFG', 'd2': {'l': [9, 10], 'b': False}, 'xyz': 123, 'c': 'black'}]
            ```
        """
        return self.apply(
            "move",
            path=path,
            new_path=new_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def rename(
        self: KnowledgeAssetT,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Rename data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.RenameAssetFunc`.

        Same as `KnowledgeAsset.move` but must specify new token instead of new path.

        Usage:
            ```pycon
            >>> asset.rename("d2.l", "x").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'x': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'x': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'x': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'x': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'x': [9, 10]}, 'xyz': 123}]

            >>> asset.rename("xyz", "zyx", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'zyx': 123}]
            ```
        """
        return self.apply(
            "rename",
            path=path,
            new_token=new_token,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def reorder(
        self: KnowledgeAssetT,
        new_order: tp.Union[str, tp.PathKeyTokens],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Reorder data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.ReorderAssetFunc`.

        Can change order in dicts based on `vectorbtpro.utils.config.reorder_dict` and
        sequences based on `vectorbtpro.utils.config.reorder_list`.

        Argument `new_order` can be a sequence of tokens. To not reorder a subset of keys, they can
        be replaced by an ellipsis (`...`). For example, `["a", ..., "z"]` puts the token "a" at the start
        and the token "z" at the end while other tokens are left in the original order. If `new_order` is
        a string, it can be "asc"/"ascending" or "desc"/"descending". Other than that, it can be a string or
        function (will become a template), or any custom template. In this template, the data item is
        the index of the data item is represented by "i", the data item itself is represented by "d",
        the data item under the path is represented by "x" while its fields are represented by their names.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to template substitution in `new_order`.

        Usage:
            ```pycon
            >>> asset.reorder(["xyz", ...], skip_missing=True).get()
            >>> asset.reorder(lambda x: ["xyz", ...] if "xyz" in x else [...]).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'xyz': 123, 's': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}}]

            >>> asset.reorder("descending", path="d2.l").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [2, 1]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [4, 3]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [6, 5]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [8, 7]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [10, 9]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "reorder",
            new_order=new_order,
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            template_context=template_context,
            **kwargs,
        )

    def query(
        self: KnowledgeAssetT,
        expression: tp.Union[str, tp.Callable, tp.CustomTemplate],
        query_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Query using an engine and return the queried data item(s).

        Following engines are supported:

        * "jmespath": Evaluation with `jmespath` package
        * "jsonpath", "jsonpath-ng" or "jsonpath_ng": Evaluation with `jsonpath-ng` package
        * "jsonpath.ext", "jsonpath-ng.ext" or "jsonpath_ng.ext": Evaluation with extended `jsonpath-ng` package
        * None or "template": Evaluation of each data item as a template. The index of the data item is
            represented by "i", the data item itself is represented by "d", the data item under the path
            is represented by "x" while its fields are represented by their names.
            Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.QueryAssetFunc`.
        * "pandas": Same as above but variables being columns

        Templates can also use the functions defined in `vectorbtpro.utils.search.search_config`.

        They work on single values and sequences alike.

        Keyword arguments are passed to the respective search/parse/evaluation function.

        Usage:
            ```pycon
            >>> asset.query("d['s'] == 'ABC'")
            >>> asset.query("x['s'] == 'ABC'")
            >>> asset.query("s == 'ABC'")
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.query("x['s'] == 'ABC'", return_type="bool")
            [True, False, False, False, False]

            >>> asset.query("find('BC', s)")
            >>> asset.query(lambda s: "BC" in s)
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.query("[?contains(s, 'BC')].s", query_engine="jmespath")
            ['ABC', 'BCD']

            >>> asset.query("[].d2.c", query_engine="jmespath")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("[?d2.c != `blue`].d2.l", query_engine="jmespath")
            [[1, 2], [5, 6], [7, 8], [9, 10]]

            >>> asset.query("$[*].d2.c", query_engine="jsonpath.ext")
            ['red', 'blue', 'green', 'yellow', 'black']

            >>> asset.query("$[?(@.b == true)].s", query_engine="jsonpath.ext")
            ['ABC', 'BCD']

            >>> asset.query("s[b]", query_engine="pandas")
            ['ABC', 'BCD']
            ```
        """
        query_engine = self.resolve_setting(query_engine, "query_engine")
        template_context = self.resolve_setting(template_context, "template_context", merge=True)
        return_type = self.resolve_setting(return_type, "return_type")

        if query_engine is None or query_engine.lower() == "template":
            new_obj = self.apply(
                "query",
                expression=expression,
                template_context=template_context,
                return_type=return_type,
                **kwargs,
            )
        elif query_engine.lower() == "jmespath":
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jmespath")
            import jmespath

            new_obj = jmespath.search(expression, self.data, **kwargs)
        elif query_engine.lower() in ("jsonpath", "jsonpath-ng", "jsonpath_ng"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng

            jsonpath_expr = jsonpath_ng.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.data, **kwargs)]
        elif query_engine.lower() in ("jsonpath.ext", "jsonpath-ng.ext", "jsonpath_ng.ext"):
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("jsonpath_ng")
            import jsonpath_ng.ext

            jsonpath_expr = jsonpath_ng.ext.parse(expression)
            new_obj = [match.value for match in jsonpath_expr.find(self.data, **kwargs)]
        elif query_engine.lower() == "pandas":
            if isinstance(expression, str):
                expression = RepEval(expression)
            elif checks.is_function(expression):
                if checks.is_builtin_func(expression):
                    expression = RepFunc(lambda _expression=expression: _expression)
                else:
                    expression = RepFunc(expression)
            elif not isinstance(expression, CustomTemplate):
                raise TypeError(f"Expression must be a template")
            df = pd.DataFrame.from_records(self.data)
            _template_context = flat_merge_dicts(
                {
                    "d": df,
                    "x": df,
                    **df.to_dict(orient="series"),
                },
                template_context,
            )
            result = expression.substitute(_template_context, eval_id="expression", **kwargs)
            if checks.is_function(result):
                result = result(df)
            if return_type is None:
                as_filter = True
            elif return_type.lower() == "bool":
                as_filter = False
            else:
                raise ValueError(f"Invalid return type: '{return_type}'")
            if as_filter and isinstance(result, pd.Series) and result.dtype == "bool":
                result = df[result]
            if isinstance(result, pd.Series):
                new_obj = result.tolist()
            elif isinstance(result, pd.DataFrame):
                new_obj = result.to_dict(orient="records")
            else:
                new_obj = result
        else:
            raise ValueError(f"Invalid query engine: '{query_engine}'")
        return new_obj

    def filter(self: KnowledgeAssetT, *args, **kwargs) -> KnowledgeAssetT:
        """Call `KnowledgeAsset.query` and return a new `KnowledgeAsset` instance."""
        return self.query(*args, wrap=True, **kwargs)

    def find(
        self: KnowledgeAssetT,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        in_dumps: tp.Optional[bool] = None,
        dump_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        return_path: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Find occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.FindAssetFunc`.

        Uses `vectorbtpro.utils.search.contains_in_obj` (keyword arguments are passed here)
        to find any occurrences in each data item if `return_type` is None (returns an entire data when match),
        `return_type` is "field" (returns the field), or `return_type` is "bool" (returns True when match).
        For all other return types, uses `vectorbtpro.utils.search.find_in_obj` and `vectorbtpro.utils.search.find`.

        Target can be one or multiple data items. If there are multiple targets and `find_all` is True,
        the match function will return True only if all targets have been found.

        Use argument `path` to specify what part of the data item should be searched. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, the data item will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the data item, will skip the data item. If `per_path` is True, will consider
        targets to be provided per path.

        Use argument `source` instead of `path` or in addition to `path` to also preprocess the source.
        It can be a string or function (will become a template), or any custom template. In this template,
        the index of the data item is represented by "i", the data item itself is represented by "d",
        the data item under the path is represented by "x" while its fields are represented by their names.

        Set `in_dumps` to True to convert the entire data item to string and search in that string.
        Will use `vectorbtpro.utils.formatting.dump` with `dump_kwargs`.

        Usage:
            ```pycon
            >>> asset.find("BC").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("BC", return_type="bool").get()
            [True, True, False, False, False]

            >>> asset.find(vbt.Not("BC")).get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find("bc", ignore_case=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find("bl", path="d2.c").get()
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(5, path="d2.l[0]").get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(True, path="d2.l", source=lambda x: sum(x) >= 10).get()
            [{'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find(["A", "B", "C"]).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find(["A", "B", "C"], find_all=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}}]

            >>> asset.find(r"[ABC]+", mode="regex").get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}}]

            >>> asset.find("yenlow", mode="fuzzy").get()
            [{'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.find("yenlow", mode="fuzzy", return_type="match").get()
            [[], [], [], ['yellow'], []]

            >>> asset.find("yenlow", mode="fuzzy", return_type="match", return_path=True).get()
            [{}, {}, {}, {('d2', 'c'): ['yellow']}, {}]

            >>> asset.find("xyz", in_dumps=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "find",
            target=target,
            path=path,
            per_path=per_path,
            find_all=find_all,
            keep_path=keep_path,
            skip_missing=skip_missing,
            source=source,
            in_dumps=in_dumps,
            dump_kwargs=dump_kwargs,
            template_context=template_context,
            return_type=return_type,
            return_path=return_path,
            **kwargs,
        )

    def find_code(
        self,
        target: tp.Optional[tp.MaybeList[tp.Any]] = None,
        language: tp.Optional[str] = None,
        in_blocks: bool = True,
        return_type: tp.Optional[str] = "match",
        flags: int = 0,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Find code using `KnowledgeAsset.find`."""
        if in_blocks and language is not None:
            raise ValueError("Language requires in_blocks=True")
        if target is not None:
            if not isinstance(target, list):
                targets = [target]
                single_target = True
            else:
                targets = target
                single_target = False
            new_target = []
            for t in targets:
                if language is not None:
                    new_t = rf"""
                    ```{re.escape(language)}\n
                    (?:(?!```)[\s\S])*?
                    {re.escape(t)}
                    (?:(?!```)[\s\S])*?
                    ```
                    """
                else:
                    if in_blocks:
                        new_t = rf"""
                        ```(?:\n)
                        (?:(?!```)[\s\S])*?
                        {re.escape(t)}
                        (?:(?!```)[\s\S])*?
                        ```
                        """
                    else:
                        new_t = rf"(?<!`)`([^`]*{re.escape(t)}[^`]*)`(?!`)"
                new_target.append(new_t)
            if single_target:
                new_target = new_target[0]
        else:
            if language is not None:
                new_target = rf"```{re.escape(language)}\n([\s\S]*?)```"
            else:
                if in_blocks:
                    new_target = r"```(?:\w+)?\n([\s\S]*?)```"
                else:
                    new_target = r"(?<!`)`([^`]*)`(?!`)"
        if in_blocks:
            flags |= re.DOTALL | re.MULTILINE | re.VERBOSE
        return self.find(new_target, mode="regex", return_type=return_type, flags=flags, **kwargs)

    def find_replace(
        self: KnowledgeAssetT,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        replacement: tp.Optional[tp.MaybeList[tp.Any]] = None,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Find and replace occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.FindReplaceAssetFunc`.

        Uses `vectorbtpro.utils.search.find_in_obj` (keyword arguments are passed here) to find
        occurrences in each data item. Then, uses `vectorbtpro.utils.search.replace_in_obj` to replace them.

        Target can be one or multiple of data items, either as a list or a dictionary. If there are multiple
        targets and `find_all` is True, the match function will return True only if all targets have been found.

        Use argument `path` to specify what part of the data item should be searched. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. If `keep_path` is True, the data item will be represented
        as a nested dictionary with path as keys. If multiple paths are provided, `keep_path` automatically
        becomes True, and they will be merged into one nested dictionary. If `skip_missing` is True
        and path is missing in the data item, will skip the data item. If `per_path` is True, will consider
        targets and replacements to be provided per path.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Usage:
            ```pycon
            >>> asset.find_replace("BC", "XY").get()
            [{'s': 'AXY', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find_replace("BC", "XY", changed_only=True).get()
            [{'s': 'AXY', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'XYD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find_replace(r"(D)E(F)", r"\1X\2", mode="regex", changed_only=True).get()
            [{'s': 'DXF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}}]

            >>> asset.find_replace(True, False, changed_only=True).get()
            [{'s': 'ABC', 'b': False, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': False, 'd2': {'c': 'blue', 'l': [3, 4]}}]

            >>> asset.find_replace(3, 30, path="d2.l", changed_only=True).get()
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [30, 4]}}]

            >>> asset.find_replace({1: 10, 4: 40}, path="d2.l", changed_only=True).get()
            >>> asset.find_replace({1: 10, 4: 40}, path=["d2.l[0]", "d2.l[1]"], changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [10, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 40]}}]

            >>> asset.find_replace({1: 10, 4: 40}, find_all=True, changed_only=True).get()
            []

            >>> asset.find_replace({1: 10, 2: 20}, find_all=True, changed_only=True).get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [10, 20]}}]

            >>> asset.find_replace("a", "X", path=["s", "d2.c"], ignore_case=True, changed_only=True).get()
            [{'s': 'XBC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'blXck', 'l': [9, 10]}, 'xyz': 123}]

            >>> asset.find_replace(123, 456, path="xyz", skip_missing=True, changed_only=True).get()
            [{'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 456}]
            ```
        """
        return self.apply(
            "find_replace",
            target=target,
            replacement=replacement,
            path=path,
            per_path=per_path,
            find_all=find_all,
            keep_path=keep_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def find_remove(
        self: KnowledgeAssetT,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Find and remove occurrences and return a new `KnowledgeAsset` instance.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.FindRemoveAssetFunc`.

        Similar to `KnowledgeAsset.find_replace`."""
        return self.apply(
            "find_remove",
            target=target,
            path=path,
            per_path=per_path,
            find_all=find_all,
            keep_path=keep_path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def find_remove_empty(self: KnowledgeAssetT, **kwargs) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Find and remove empty objects."""
        from vectorbtpro.utils.knowledge.base_asset_funcs import FindRemoveAssetFunc

        return self.find_remove(FindRemoveAssetFunc.is_empty_func, **kwargs)

    def flatten(
        self: KnowledgeAssetT,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Flatten data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.FlattenAssetFunc`.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to `vectorbtpro.utils.search.flatten_obj`.

        Usage:
            ```pycon
            >>> asset.flatten().get()
            [{'s': 'ABC',
              'b': True,
              ('d2', 'c'): 'red',
              ('d2', 'l', 0): 1,
              ('d2', 'l', 1): 2},
              ...
             {'s': 'EFG',
              'b': False,
              ('d2', 'c'): 'black',
              ('d2', 'l', 0): 9,
              ('d2', 'l', 1): 10,
              'xyz': 123}]
            ```
        """
        return self.apply(
            "flatten",
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def unflatten(
        self: KnowledgeAssetT,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Unflatten data items or parts of them.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.UnflattenAssetFunc`.

        Use argument `path` to specify what part of the data item should be set. For example, "x.y[0].z"
        to navigate nested dictionaries/lists. Multiple paths can be provided. If `skip_missing` is True and
        path is missing in the data item, will skip the data item.

        Set `make_copy` to True to not modify original data.

        Set `changed_only` to True to keep only the data items that have been changed.

        Keyword arguments are passed to `vectorbtpro.utils.search.unflatten_obj`.

        Usage:
            ```pycon
            >>> asset.flatten().unflatten().get()
            [{'s': 'ABC', 'b': True, 'd2': {'c': 'red', 'l': [1, 2]}},
             {'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'CDE', 'b': False, 'd2': {'c': 'green', 'l': [5, 6]}},
             {'s': 'DEF', 'b': False, 'd2': {'c': 'yellow', 'l': [7, 8]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        return self.apply(
            "unflatten",
            path=path,
            skip_missing=skip_missing,
            make_copy=make_copy,
            changed_only=changed_only,
            **kwargs,
        )

    def dump(
        self: KnowledgeAssetT,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        dump_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Dump data items.

        Uses `KnowledgeAsset.apply` on `vectorbtpro.utils.knowledge.base_asset_funcs.DumpAssetFunc`.

        Following engines are supported:

        * "repr": Dumping with `repr`
        * "prettify": Dumping with `vectorbtpro.utils.formatting.prettify`
        * "nestedtext": Dumping with NestedText (https://pypi.org/project/nestedtext/)
        * "yaml": Dumping with YAML
        * "toml": Dumping with TOML (https://pypi.org/project/toml/)
        * "json": Dumping with JSON

        Use argument `source` to also preprocess the source. It can be a string or function
        (will become a template), or any custom template. In this template, the index of the data item
        is represented by "i", the data item itself is represented by "d" while its fields are
        represented by their names.

        Keyword arguments are passed to the respective engine.

        Usage:
            ```pycon
            >>> print(asset.dump(source="{i: d}", default_flow_style=True).join())
            {0: {s: ABC, b: true, d2: {c: red, l: [1, 2]}}}
            {1: {s: BCD, b: true, d2: {c: blue, l: [3, 4]}}}
            {2: {s: CDE, b: false, d2: {c: green, l: [5, 6]}}}
            {3: {s: DEF, b: false, d2: {c: yellow, l: [7, 8]}}}
            {4: {s: EFG, b: false, d2: {c: black, l: [9, 10]}, xyz: 123}}
            ```
        """
        return self.apply(
            "dump",
            source=source,
            dump_engine=dump_engine,
            template_context=template_context,
            **kwargs,
        )

    def dump_all(
        self,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        dump_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> str:
        """Dump data list as a single data item.

        See `KnowledgeAsset.dump` for arguments."""
        from vectorbtpro.utils.knowledge.base_asset_funcs import DumpAssetFunc

        return DumpAssetFunc.prepare_and_call(
            self.data,
            source=source,
            dump_engine=dump_engine,
            template_context=template_context,
            **kwargs,
        )

    # ############# Reduce methods ############# #

    @classmethod
    def get_keys_and_groups(
        cls,
        by: tp.List[tp.Any],
        uniform_groups: bool = False,
    ) -> tp.Tuple[tp.List[tp.Any], tp.List[tp.List[int]]]:
        """get keys and groups."""
        keys = []
        groups = []
        if uniform_groups:
            for i, item in enumerate(by):
                if len(keys) > 0 and (keys[-1] is item or keys[-1] == item):
                    groups[-1].append(i)
                else:
                    keys.append(item)
                    groups.append([i])
        else:
            groups = []
            representatives = []
            for idx, item in enumerate(by):
                found = False
                for rep_idx, rep_obj in enumerate(representatives):
                    if item is rep_obj or item == rep_obj:
                        groups[rep_idx].append(idx)
                        found = True
                        break
                if not found:
                    representatives.append(item)
                    keys.append(by[idx])
                    groups.append([idx])
        return keys, groups

    def reduce(
        self: KnowledgeAssetT,
        func: tp.Union[str, tp.Callable, tp.CustomTemplate],
        *args,
        initializer: tp.Optional[tp.Any] = None,
        by: tp.Optional[tp.PathLikeKey] = None,
        template_context: tp.KwargsLike = None,
        show_progress: tp.Optional[bool] = None,
        pbar_kwargs: tp.KwargsLike = None,
        wrap: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Reduce data items.

        Function can be a callable, a tuple of function and its arguments,
        a `vectorbtpro.utils.execution.Task` instance, a subclass of
        `vectorbtpro.utils.knowledge.base_asset_funcs.AssetFunc` or its prefix or full name.
        It can also be an expression or a template. In this template, the index of the data item is
        represented by "i", the data items themselves are represented by "d1" and "d2" or "x1" and "x2".

        If an initializer is provided, the first set of values will be `d1=initializer` and
        `d2=self.data[0]`. If not, it will be `d1=self.data[0]` and `d2=self.data[1]`.

        If `by` is provided, see `KnowledgeAsset.groupby_reduce`.

        If `wrap` is True, returns a new `KnowledgeAsset` instance, otherwise raw output.

        Positional arguments are passed to the function in case the template returns another function,
        and keyword arguments are passed as a context to template substitution.

        Usage:
            ```pycon
            >>> asset.reduce(lambda d1, d2: vbt.merge_dicts(d1, d2))
            >>> asset.reduce(vbt.merge_dicts)
            >>> asset.reduce("{**d1, **d2}")
            {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}

            >>> asset.reduce("{**d1, **d2}", by="b")
            [{'s': 'BCD', 'b': True, 'd2': {'c': 'blue', 'l': [3, 4]}},
             {'s': 'EFG', 'b': False, 'd2': {'c': 'black', 'l': [9, 10]}, 'xyz': 123}]
            ```
        """
        if by is not None:
            return self.groupby_reduce(
                func,
                *args,
                by=by,
                initializer=initializer,
                template_context=template_context,
                show_progress=show_progress,
                pbar_kwargs=pbar_kwargs,
                wrap=wrap,
                **kwargs,
            )

        show_progress = self.resolve_setting(show_progress, "show_progress")
        pbar_kwargs = self.resolve_setting(pbar_kwargs, "pbar_kwargs", merge=True)

        asset_func_meta = {}

        if isinstance(func, str) and not func.isidentifier():
            func = RepEval(func)
        elif not isinstance(func, CustomTemplate):
            from vectorbtpro.utils.knowledge.asset_pipelines import AssetPipeline

            func, args, kwargs = AssetPipeline.resolve_task(
                func,
                *args,
                cond_kwargs=dict(asset=self),
                asset_func_meta=asset_func_meta,
                **kwargs,
            )

        it = iter(self.data)
        if initializer is None and asset_func_meta.get("_initializer", None) is not None:
            initializer = asset_func_meta["_initializer"]
        if initializer is None:
            d1 = next(it)
            total = len(self.data) - 1
            if total == 0:
                raise ValueError("Must provide initializer")
        else:
            d1 = initializer
            total = len(self.data)
        if show_progress is None:
            show_progress = total > 1
        prefix = get_caller_qualname().split(".")[-1]
        if "_short_name" in asset_func_meta:
            prefix += f"[{asset_func_meta['_short_name']}]"
        elif isinstance(func, type):
            prefix += f"[{func.__name__}]"
        else:
            prefix += f"[{type(func).__name__}]"
        pbar_kwargs = flat_merge_dicts(
            dict(
                bar_id=get_caller_qualname(),
                prefix=prefix,
            ),
            pbar_kwargs,
        )
        with ProgressBar(total=total, show_progress=show_progress, **pbar_kwargs) as pbar:
            for i, d2 in enumerate(it):
                if isinstance(func, CustomTemplate):
                    _template_context = flat_merge_dicts(
                        {
                            "i": i,
                            "d1": d1,
                            "d2": d2,
                            "x1": d1,
                            "x2": d2,
                        },
                        template_context,
                    )
                    _d1 = func.substitute(_template_context, eval_id="func", **kwargs)
                    if checks.is_function(_d1):
                        d1 = _d1(d1, d2, *args)
                    else:
                        d1 = _d1
                else:
                    _kwargs = dict(kwargs)
                    if "template_context" in _kwargs:
                        _kwargs["template_context"] = flat_merge_dicts(
                            {"i": i},
                            _kwargs["template_context"],
                        )
                    d1 = func(d1, d2, *args, **_kwargs)
                pbar.update()
        if wrap is None and asset_func_meta.get("_wrap", None) is not None:
            wrap = asset_func_meta["_wrap"]
        if wrap is None:
            wrap = False
        if wrap:
            if not isinstance(d1, list):
                d1 = [d1]
            return self.replace(data=d1, single_item=True)
        return d1

    def groupby_reduce(
        self: KnowledgeAssetT,
        func: tp.Union[str, tp.Callable, tp.CustomTemplate],
        *args,
        by: tp.Optional[tp.PathLikeKey] = None,
        uniform_groups: tp.Optional[bool] = None,
        get_kwargs: tp.KwargsLike = None,
        execute_kwargs: tp.KwargsLike = None,
        return_group_keys: bool = False,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, dict, list]:
        """Group data items by keys and reduce.

        If `by` is provided, uses it as `path` in `KnowledgeAsset.get`, groups by unique values,
        and runs `KnowledgeAsset.reduce` on each group.

        Set `uniform_groups` to True to only group unique values that are located adjacent to each other.

        Variable arguments are passed to each call of `KnowledgeAsset.reduce`."""
        uniform_groups = self.resolve_setting(uniform_groups, "uniform_groups")
        execute_kwargs = self.resolve_setting(execute_kwargs, "execute_kwargs", merge=True)

        if get_kwargs is None:
            get_kwargs = {}
        by = self.get(path=by, **get_kwargs)
        keys, groups = self.get_keys_and_groups(by, uniform_groups=uniform_groups)
        if len(groups) == 0:
            raise ValueError("Groups are empty")
        tasks = []
        for i, group in enumerate(groups):
            group_instance = self.get_item(group)
            tasks.append(Task(group_instance.reduce, func, *args, **kwargs))
        prefix = get_caller_qualname().split(".")[-1]
        execute_kwargs = deep_merge_dicts(
            dict(
                show_progress=len(groups) > 1,
                pbar_kwargs=dict(
                    bar_id=get_caller_qualname(),
                    prefix=prefix,
                ),
            ),
            execute_kwargs,
        )
        results = execute(tasks, size=len(groups), **execute_kwargs)
        if return_group_keys:
            return dict(zip(keys, results))
        if len(results) > 0 and isinstance(results[0], type(self)):
            return type(self).stack(results)
        return results

    def merge_dicts(self: KnowledgeAssetT, **kwargs) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Merge (dict) date items into a single dict.

        Final keyword arguments are passed to `vectorbtpro.utils.config.merge_dicts`."""
        return self.reduce("merge_dicts", **kwargs)

    def merge_lists(self: KnowledgeAssetT, **kwargs) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Merge (list) date items into a single list."""
        return self.reduce("merge_lists", **kwargs)

    def collect(
        self: KnowledgeAssetT,
        sort_keys: tp.Optional[bool] = None,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, tp.Any]:
        """Collect values of each key in each data item."""
        return self.reduce("collect", sort_keys=sort_keys, **kwargs)

    @classmethod
    def describe_lengths(self, lengths: list, **describe_kwargs) -> dict:
        """Describe values representing lengths."""
        len_describe_dict = pd.Series(lengths).describe(**describe_kwargs).to_dict()
        del len_describe_dict["count"]
        del len_describe_dict["std"]
        return {"len_" + k: int(v) if k != "mean" else v for k, v in len_describe_dict.items()}

    def describe(
        self: KnowledgeAssetT,
        ignore_empty: tp.Optional[bool] = None,
        describe_kwargs: tp.KwargsLike = None,
        wrap: bool = False,
        **kwargs,
    ) -> tp.Union[KnowledgeAssetT, dict]:
        """Collect and describe each key in each data item."""
        ignore_empty = self.resolve_setting(ignore_empty, "ignore_empty")
        describe_kwargs = self.resolve_setting(describe_kwargs, "describe_kwargs", merge=True)

        collected = self.collect(**kwargs)
        description = {}
        for k, v in list(collected.items()):
            all_types = []
            valid_types = []
            valid_x = None
            new_v = []
            for x in v:
                if not ignore_empty or x:
                    new_v.append(x)
                if x is not None:
                    valid_x = x
                    if type(x) not in valid_types:
                        valid_types.append(type(x))
                if type(x) not in all_types:
                    all_types.append(type(x))
            v = new_v
            description[k] = {}
            description[k]["types"] = list(map(lambda x: x.__name__, all_types))
            describe_sr = pd.Series(v)
            if describe_sr.dtype == object and len(valid_types) == 1 and checks.is_complex_collection(valid_x):
                describe_dict = {"count": len(v)}
            else:
                describe_dict = describe_sr.describe(**describe_kwargs).to_dict()
            if pd.api.types.is_integer_dtype(describe_sr.dtype):
                new_describe_dict = {}
                for _k, _v in describe_dict.items():
                    if _k not in {"mean", "std"}:
                        new_describe_dict[_k] = int(_v)
                    else:
                        new_describe_dict[_k] = _v
                describe_dict = new_describe_dict
            if "unique" in describe_dict and describe_dict["unique"] == describe_dict["count"]:
                del describe_dict["top"]
                del describe_dict["freq"]
            if "unique" in describe_dict and describe_dict["count"] == 1:
                del describe_dict["unique"]
            description[k].update(describe_dict)
            if len(valid_types) == 1 and checks.is_collection(valid_x):
                lengths = [len(_v) for _v in v if _v is not None]
                description[k].update(self.describe_lengths(lengths, **describe_kwargs))
        if wrap:
            return self.replace(data=[description], single_item=True)
        return description

    def print_schema(self, **kwargs) -> None:
        """Print schema.

        Keyword arguments are split between `KnowledgeAsset.describe` and
        `vectorbtpro.utils.path_.dir_tree_from_paths`.

        Usage:
            ```pycon
            >>> asset.print_schema()
            /
            ├── s [5/5, str]
            ├── b [2/5, bool]
            ├── d2 [5/5, dict]
            │   ├── c [5/5, str]
            │   └── l
            │       ├── 0 [5/5, int]
            │       └── 1 [5/5, int]
            └── xyz [1/5, int]

            2 directories, 6 files
            ```
        """
        dir_tree_arg_names = set(get_func_arg_names(dir_tree_from_paths))
        dir_tree_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in dir_tree_arg_names}
        orig_describe_dict = self.describe(wrap=False, **kwargs)
        flat_describe_dict = self.flatten(
            skip_missing=True,
            make_copy=True,
            changed_only=False,
        ).describe(wrap=False, **kwargs)
        describe_dict = flat_merge_dicts(orig_describe_dict, flat_describe_dict)
        paths = []
        path_names = []
        for k, v in describe_dict.items():
            if k is None:
                k = "."
            if not isinstance(k, tuple):
                k = (k,)
            path = Path(*map(str, k))
            path_name = path.name
            path_name += " [" + str(v["count"]) + "/" + str(len(self.data))
            path_name += ", " + ", ".join(v["types"]) + "]"
            path_names.append(path_name)
            paths.append(path)
        if "root_name" not in dir_tree_kwargs:
            dir_tree_kwargs["root_name"] = "/"
        if "sort" not in dir_tree_kwargs:
            dir_tree_kwargs["sort"] = False
        if "path_names" not in dir_tree_kwargs:
            dir_tree_kwargs["path_names"] = path_names
        if "length_limit" not in dir_tree_kwargs:
            dir_tree_kwargs["length_limit"] = None
        print(dir_tree_from_paths(paths, **dir_tree_kwargs))

    def join(self, separator: tp.Optional[str] = None) -> str:
        """Join the list of string data items."""
        if len(self.data) == 1:
            return self.data[0]
        if separator is None:
            if not all(isinstance(d, str) for d in self.data):
                raise TypeError("All data items must be strings")
            if self.data[0].endswith(("\n", "\t", " ")):
                separator = ""
            elif self.data[0].endswith(("}", "]")):
                separator = ", "
            else:
                separator = "\n"
        joined = separator.join(self.data)
        if joined.startswith("{") and joined.endswith("}"):
            return "[" + joined + "]"
        return joined

    def to_context(
        self,
        *args,
        dump_all: tp.Optional[bool] = None,
        separator: tp.Optional[str] = None,
        **kwargs,
    ) -> str:
        """Dump and join to context."""
        if dump_all is None:
            dump_all = len(self.data) > 1 and separator is None
        if dump_all:
            return self.dump_all(*args, **kwargs)
        dumped = self.dump(*args, **kwargs)
        if isinstance(dumped, KnowledgeAsset):
            return dumped.join(separator=separator)
        return dumped

    def print(self, *args, **kwargs) -> None:
        """Convert to context and print.

        Uses `KnowledgeAsset.to_context`."""
        print(self.to_context(*args, **kwargs))

    # ############# LLM methods ############# #

    def chat(
        self,
        question: str,
        chat_history: tp.Optional[tp.MutableSequence[str]] = None,
        stream: tp.Optional[bool] = None,
        to_context_kwargs: tp.KwargsLike = None,
        max_context_chars: tp.Optional[int] = None,
        max_context_tokens: tp.Optional[int] = None,
        tokenizer: tp.Union[None, str, EncodingT] = None,
        system_prompt: tp.Optional[str] = None,
        context_prompt: tp.Optional[str] = None,
        display_format: tp.Optional[str] = None,
        refresh_rate: tp.Optional[float] = None,
        to_markdown_kwargs: tp.KwargsLike = None,
        to_html_kwargs: tp.KwargsLike = None,
        format_html_kwargs: tp.KwargsLike = None,
        open_browser: tp.Optional[bool] = None,
        cache: tp.Optional[bool] = None,
        cache_dir: tp.Optional[tp.PathLike] = None,
        cache_mkdir_kwargs: tp.KwargsLike = None,
        clear_cache: bool = False,
        file_prefix_len: tp.Optional[int] = None,
        file_suffix_len: tp.Optional[int] = None,
        output_to: tp.Optional[tp.Union[str, tp.TextIO]] = None,
        flush_output: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        package: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.Optional[Path]:
        """Chat with an LLM using LlamaIndex and the dumped asset as a context.

        The following packages are supported:

        * "openai"
        * "litellm"
        * "llama_index"

        For LlamaIndex, LLM can be provided via `llm`, which can be either the name of the class,
        the path to the class (accepted are both "llama_index.xxx.yyy" and "xxx.yyy"), or a subclass
        or an instance of `llama_index.core.llms.LLM`. Case of strings doesn't matter.

        Uses `context_prompt` as a prompt template requiring the variable "context".
        The prompt can be either a custom template, or string or function that will become one.
        The context itself is generated by dumping and joining data items with `KnowledgeAsset.to_context`
        with `to_context_kwargs` as keyword arguments. Once the prompt is evaluated, it becomes a system message.
        Uses `system_prompt` as a system prompt following the context prompt.

        Use `max_context_chars` to limit the number of characters in the context. Use `max_context_tokens`
        to limit the number of tokens in the context (requires tiktoken). Use `tokenizer` to provide
        a model name, an encoding name, or an encoding object for tokenization.

        Pass `chat_history` as a mutable sequence (for example, list) to keep track of chat history.
        After generating a response, the output will be appended to this sequence as an assistant message.

        Argument `question` becomes a user message.

        If the output is a stream (`stream=True`), appends chunks one by one and displays the
        intermediate result. Otherwise, displays the entire message.

        The following display formats are supported:

        * "plain": Uses `print` to print the output in its raw format
        * "ipython_markdown": Uses `IPython` to render the output as a Markdown in a notebook environment
        * "ipython_html": Uses `IPython` to render the output as an HTML in a notebook environment
        * "html": Renders the output as a static HTML page and displays it in the browser
        * "auto_ipython": Decides between using "ipython_html" or "plain" depending on the environment

        If `refresh_rate` is None, will refresh as soon as the next chunk arrives (apart from refreshing
        HTML pages, here the minimum refresh rate is 1 second). Otherwise, will collect chunks and
        refresh once in `refresh_rate` seconds.

        To convert to Markdown, uses
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToMarkdownAssetFunc.to_markdown`
        along with `to_markdown_kwargs`. To convert to HTML, uses
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.to_html`
        along with `to_html_kwargs`. To create and format the HTML page, uses
        `vectorbtpro.utils.knowledge.custom_asset_funcs.ToHTMLAssetFunc.format_html`
        along with `format_html_kwargs`.

        The HTML file is stored either in the cache directory (`cache=True`)
        under "chat" and with truncated title as prefix and random hash as suffix, or as a
        temporary file (`cache=False`). If `clear_cache` is True, deletes any existing directory
        before creating a new one. Returns the path of the directory where HTML file is stored.

        The raw output can be also redirected to a file (or any IO) specified in `output_to`
        and flushed with each chunk if `flush_output` is True.

        Keyword arguments are distributed between the package client (if exists) and completion API.

        For defaults, see `chat` in `vectorbtpro._settings.knowledge`.

        Usage:
            ```pycon
            >>> asset.chat("What's the value under 'xyz'?")
            The value under 'xyz' is 123.

            >>> chat_history = []
            >>> asset.chat("What's the value under 'xyz'?", chat_history=chat_history)
            The value under 'xyz' is 123.

            >>> asset.chat("Are you sure?", chat_history=chat_history)
            Yes, I am sure. The value under 'xyz' is 123 for the entry where `s` is "EFG".
            ```
        """
        stream = self.resolve_setting(stream, "stream", sub_path="chat")
        to_context_kwargs = self.resolve_setting(to_context_kwargs, "to_context_kwargs", merge=True, sub_path="chat")
        max_context_chars = self.resolve_setting(max_context_chars, "max_context_chars", sub_path="chat")
        max_context_tokens = self.resolve_setting(max_context_tokens, "max_context_tokens", sub_path="chat")
        tokenizer = self.resolve_setting(tokenizer, "tokenizer", sub_path="chat")
        system_prompt = self.resolve_setting(system_prompt, "system_prompt", sub_path="chat")
        context_prompt = self.resolve_setting(context_prompt, "context_prompt", sub_path="chat")
        display_format = self.resolve_setting(display_format, "display_format", sub_path="chat")
        refresh_rate = self.resolve_setting(refresh_rate, "refresh_rate", sub_path="chat")
        to_markdown_kwargs = self.resolve_setting(to_markdown_kwargs, "to_markdown_kwargs", merge=True, sub_path="chat")
        to_html_kwargs = self.resolve_setting(to_html_kwargs, "to_html_kwargs", merge=True, sub_path="chat")
        format_html_kwargs = self.resolve_setting(format_html_kwargs, "format_html_kwargs", merge=True, sub_path="chat")
        open_browser = self.resolve_setting(open_browser, "open_browser", sub_path="chat")
        cache = self.resolve_setting(cache, "cache", sub_path="chat")
        cache_dir = self.resolve_setting(cache_dir, "cache_dir", sub_path="chat")
        cache_mkdir_kwargs = self.resolve_setting(cache_mkdir_kwargs, "cache_mkdir_kwargs", merge=True, sub_path="chat")
        file_prefix_len = self.resolve_setting(file_prefix_len, "file_prefix_len", sub_path="chat")
        file_suffix_len = self.resolve_setting(file_suffix_len, "file_suffix_len", sub_path="chat")
        output_to = self.resolve_setting(output_to, "output_to", sub_path="chat")
        flush_output = self.resolve_setting(flush_output, "flush_output", sub_path="chat")
        template_context = self.resolve_setting(template_context, "template_context", merge=True, sub_path="chat")
        package = self.resolve_setting(package, "package", sub_path="chat")

        if chat_history is None:
            chat_history = []
        if isinstance(output_to, (str, Path)):
            output_to = Path(output_to).open("w")
            close_handle = True
        else:
            close_handle = False
        if isinstance(display_format, str):
            if display_format.lower() == "auto_ipython":
                if checks.in_notebook():
                    display_format = "ipython_html"
                else:
                    display_format = "plain"
            elif display_format.lower() not in {"plain", "ipython_markdown", "ipython_html", "html"}:
                raise ValueError(f"Invalid output format: '{display_format}'")
        if isinstance(context_prompt, str):
            context_prompt = Sub(context_prompt)
        elif checks.is_function(context_prompt):
            context_prompt = RepFunc(context_prompt)
        elif not isinstance(context_prompt, CustomTemplate):
            raise TypeError(f"Context prompt must be a string, function, or template")
        context = self.to_context(**to_context_kwargs)
        if max_context_chars is not None:
            context = context[:max_context_chars]
        if max_context_tokens is not None:
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("tiktoken")
            if isinstance(tokenizer, str):
                from tiktoken.model import MODEL_TO_ENCODING

                if tokenizer in MODEL_TO_ENCODING.keys():
                    from tiktoken import encoding_for_model

                    tokenizer = encoding_for_model(tokenizer)
                else:
                    from tiktoken import get_encoding

                    tokenizer = get_encoding(tokenizer)
            context = tokenizer.decode(tokenizer.encode(context)[:max_context_tokens])
        template_context = flat_merge_dicts(dict(context=context), template_context)
        context_prompt = context_prompt.substitute(template_context, eval_id="context_prompt")
        messages = [
            dict(role="system", content=system_prompt),
            dict(role="user", content=context_prompt),
            *chat_history,
            dict(role="user", content=question),
        ]

        if package is None:
            from vectorbtpro.utils.module_ import check_installed

            if check_installed("openai"):
                package = "openai"
            elif check_installed("litellm"):
                package = "litellm"
            elif check_installed("llama_index"):
                package = "llama_index"
            else:
                raise ValueError("No LLM packages installed")
        if package.lower() == "openai":
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("openai")
            from openai import OpenAI

            openai_config = self.resolve_setting(kwargs, "openai_config", merge=True, sub_path="chat")
            model = openai_config.pop("model", None)
            if model is None:
                raise ValueError("Must provide a model")
            client_arg_names = set(get_func_arg_names(OpenAI.__init__))
            client_kwargs = {}
            chat_kwargs = {}
            for k, v in openai_config.items():
                if k in client_arg_names:
                    client_kwargs[k] = v
                else:
                    chat_kwargs[k] = v
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(messages=messages, model=model, stream=stream, **chat_kwargs)

            def _get_chunk_content(response_chunk):
                return response_chunk.choices[0].delta.content

            def _get_full_content(response):
                return response.choices[0].message.content

        elif package.lower() == "litellm":
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("litellm")
            from litellm import completion

            litellm_config = self.resolve_setting(kwargs, "litellm_config", merge=True, sub_path="chat")
            model = litellm_config.pop("model", None)
            if model is None:
                raise ValueError("Must provide a model")
            response = completion(messages=messages, model=model, stream=stream, **litellm_config)

            def _get_chunk_content(response_chunk):
                return response_chunk.choices[0].delta.content

            def _get_full_content(response):
                return response.choices[0].message.content

        elif package.lower() == "llama_index":
            from vectorbtpro.utils.module_ import assert_can_import

            assert_can_import("llama_index")
            from llama_index.core.llms import LLM, ChatMessage

            llama_index_config = self.resolve_setting(kwargs, "llama_index_config", merge=True, sub_path="chat")
            llm = llama_index_config.pop("llm", None)
            if llm is None:
                raise ValueError("Must provide an LLM name or path")
            if isinstance(llm, str):
                from importlib import import_module
                from vectorbtpro.utils.module_ import search_package

                if "." in llm:
                    path_parts = llm.split(".")
                    llm = path_parts[-1]
                    if len(path_parts) == 2:
                        module_name = "llama_index.llms." + path_parts[0]
                    else:
                        module_name = ".".join(path_parts[:-1])
                    module = import_module(module_name)
                else:
                    import llama_index.llms

                    module = llama_index.llms
                match_func = lambda k, v: isinstance(v, type) and issubclass(v, LLM)
                candidates = search_package(module, match_func)
                class_found = False
                for k, v in candidates.items():
                    if llm.lower().replace("_", "") == k.lower():
                        llm = v
                        class_found = True
                        break
                if not class_found:
                    raise ValueError(f"LLM '{llm}' not found")
            if isinstance(llm, type):
                llm_name = llm.__name__.lower()
                module_name = llm.__module__
            else:
                checks.assert_instance_of(llm, LLM, arg_name="llm")
                llm_name = type(llm).__name__.lower()
                module_name = type(llm).__module__
            llm_configs = llama_index_config.pop("llm_configs", {})
            if llm_name in llm_configs:
                llama_index_config = deep_merge_dicts(llama_index_config, llm_configs[llm_name])
            elif module_name in llm_configs:
                llama_index_config = deep_merge_dicts(llama_index_config, llm_configs[module_name])
            if isinstance(llm, type):
                llm = llm(**llama_index_config)
            elif len(kwargs) > 0:
                raise ValueError("Cannot apply config to already initialized LLM")
            messages = list(map(lambda x: ChatMessage(**dict(x)), messages))
            if stream:
                response = llm.stream_chat(messages)
            else:
                response = llm.chat(messages)

            def _get_chunk_content(response_chunk):
                return response_chunk.delta

            def _get_full_content(response):
                return response.message.content

        else:
            raise ValueError(f"Invalid package: '{package}'")

        markdown_output = None
        html_output = None
        file_path = None

        def _display_ipython_markdown(output_content):
            nonlocal markdown_output

            from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc

            assert_can_import("IPython")
            from IPython.display import display, Markdown

            if markdown_output is None:
                markdown_output = display("", display_id=True)
            markdown_content = ToMarkdownAssetFunc.to_markdown(output_content, **to_markdown_kwargs)
            markdown_output.update(Markdown(markdown_content))

        def _display_ipython_html(output_content):
            nonlocal html_output

            from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc, ToHTMLAssetFunc

            assert_can_import("IPython")
            from IPython.display import display, HTML

            if html_output is None:
                html_output = display("", display_id=True)

            markdown_content = ToMarkdownAssetFunc.to_markdown(output_content, **to_markdown_kwargs)
            html_content = ToHTMLAssetFunc.to_html(markdown_content, **to_html_kwargs)
            html_output.update(HTML(html_content))

        def _generate_html_filename(title):
            import secrets
            import string

            title = title.lower().replace(" ", "-")
            if len(title) > file_prefix_len:
                words = title.split("-")
                truncated_title = ""
                for word in words:
                    if len(truncated_title) + len(word) + 1 <= file_prefix_len:
                        truncated_title += word + "-"
                    else:
                        break
                truncated_title = truncated_title.rstrip("-")
            else:
                truncated_title = title
            suffix_chars = string.ascii_lowercase + string.digits
            random_suffix = "".join(secrets.choice(suffix_chars) for _ in range(file_suffix_len))
            short_filename = f"{truncated_title}-{random_suffix}.html"
            return short_filename

        def _display_html(output_content, stream):
            nonlocal file_path

            from vectorbtpro.utils.knowledge.custom_asset_funcs import ToMarkdownAssetFunc, ToHTMLAssetFunc
            import tempfile

            markdown_content = ToMarkdownAssetFunc.to_markdown(output_content, **to_markdown_kwargs)
            html_content = ToHTMLAssetFunc.to_html(markdown_content, **to_html_kwargs)
            if stream:
                refresh_content = max(1, int(refresh_rate)) if refresh_rate is not None else 1
                _format_html_kwargs = dict(format_html_kwargs)
                head_extras = _format_html_kwargs["head_extras"]
                if head_extras is None:
                    head_extras = []
                if isinstance(head_extras, str):
                    head_extras = [head_extras]
                else:
                    head_extras = list(head_extras)
                head_extras.insert(0, f'<meta http-equiv="refresh" content="{refresh_content}">')
                _format_html_kwargs["head_extras"] = head_extras
                html_content = '<div id="overlay" class="overlay"></div>\n' + html_content
            else:
                _format_html_kwargs = format_html_kwargs
            html = ToHTMLAssetFunc.format_html(
                title=question,
                html_content=html_content,
                **_format_html_kwargs,
            )
            if file_path is None:
                if cache:
                    html_dir = Path(cache_dir) / "chat"
                    if html_dir.exists():
                        if clear_cache:
                            remove_dir(html_dir, missing_ok=True, with_contents=True)
                    check_mkdir(html_dir, **cache_mkdir_kwargs)
                    file_name = _generate_html_filename(question)
                    file_path = html_dir / file_name
                    with open(str(file_path.resolve()), "w", encoding="utf-8") as f:
                        f.write(html)
                else:
                    with tempfile.NamedTemporaryFile(
                        "w",
                        encoding="utf-8",
                        prefix=get_caller_qualname() + "_",
                        suffix=".html",
                        delete=False,
                    ) as f:
                        f.write(html)
                        file_path = Path(f.name)
                if open_browser:
                    import webbrowser

                    webbrowser.open("file://" + str(file_path.resolve()))
            else:
                with open(str(file_path.resolve()), "w", encoding="utf-8") as f:
                    f.write(html)

        if stream:
            buffer_contents = []
            chunk_contents = []
            in_code_block = False
            for i, response_chunk in enumerate(response):
                chunk_content = _get_chunk_content(response_chunk)
                if chunk_content is not None:
                    buffer_contents.append(chunk_content)
                    if refresh_rate is not None:
                        import time

                        if i == 0:
                            t = time.time()
                        if time.time() - t >= refresh_rate:
                            t = time.time()
                        else:
                            continue

                    if display_format.lower() == "plain" or output_to is not None:
                        print(chunk_content, end="", file=output_to, flush=flush_output)
                    else:
                        buffer_content = "".join(buffer_contents)
                        semi_full_content = "".join(chunk_contents[-2:] + buffer_contents)
                        buffer_content = semi_full_content[-len(buffer_content) - 2:]
                        lines = buffer_content.split("\n")
                        for line in lines:
                            if line.strip().startswith("```"):
                                in_code_block = not in_code_block
                    if in_code_block:
                        full_content = "".join(chunk_contents + buffer_contents + ["\n```\n"])
                    else:
                        full_content = "".join(chunk_contents + buffer_contents)
                    if display_format.lower() == "ipython_markdown":
                        _display_ipython_markdown(full_content)
                    if display_format.lower() == "ipython_html":
                        _display_ipython_html(full_content)
                    if display_format.lower() == "html":
                        _display_html(full_content, True)
                    chunk_contents.extend(buffer_contents)
                    buffer_contents = []

            full_content = "".join(chunk_contents)
            if display_format.lower() == "ipython_markdown":
                _display_ipython_markdown(full_content)
            if display_format.lower() == "ipython_html":
                _display_ipython_html(full_content)
            if display_format.lower() == "html":
                _display_html(full_content, False)
        else:
            full_content = _get_full_content(response)
            if display_format.lower() == "plain" or output_to is not None:
                print(full_content, file=output_to, flush=flush_output)
            if display_format.lower() == "ipython_markdown":
                _display_ipython_markdown(full_content)
            if display_format.lower() == "ipython_html":
                _display_ipython_html(full_content)
            if display_format.lower() == "html":
                _display_html(full_content, False)

        chat_history.append(dict(role="user", content=question))
        chat_history.append(dict(role="assistant", content=full_content))
        if close_handle:
            output_to.close()
        return file_path
