# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Base asset function classes."""

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks, search
from vectorbtpro.utils.config import reorder_dict, reorder_list
from vectorbtpro.utils.template import CustomTemplate, RepEval, RepFunc
from vectorbtpro.utils.config import merge_dicts, flat_merge_dicts, deep_merge_dicts
from vectorbtpro.utils.parsing import get_func_arg_names
from vectorbtpro.utils.execution import NoResult
from vectorbtpro.utils.formatting import dump

__all__ = [
    "AssetFunc",
]


class AssetFunc:
    """Abstract class representing an asset function."""

    _short_name: tp.ClassVar[tp.Optional[str]] = None
    """Short name of the function to be used in expressions."""

    _wrap: tp.ClassVar[tp.Optional[str]] = None
    """Whether the results are meant to be wrapped with `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset`."""

    @classmethod
    def prepare(cls, *args, **kwargs) -> tp.ArgsKwargs:
        """Prepare positional and keyword arguments."""
        return args, kwargs

    @classmethod
    def call(cls, d: tp.Any, *args, **kwargs) -> tp.Any:
        """Call the function."""
        raise NotImplementedError

    @classmethod
    def prepare_and_call(cls, d: tp.Any, *args, **kwargs):
        """Prepare arguments and call the function."""
        args, kwargs = cls.prepare(*args, **kwargs)
        return cls.call(d, *args, **kwargs)


class GetAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.get`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "get"

    _wrap: tp.ClassVar[tp.Optional[str]] = False

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if path is not None:
            if isinstance(path, list):
                path = [search.resolve_pathlike_key(p) for p in path]
            else:
                path = search.resolve_pathlike_key(path)
        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(f"Source must be a string, function, or template")
        return (), {
            **dict(
                path=path,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Optional[tp.CustomTemplate] = None,
        template_context: tp.KwargsLike = None,
    ) -> tp.Any:
        x = d
        if path is not None:
            if isinstance(path, list):
                xs = []
                for p in path:
                    try:
                        xs.append(search.get_pathlike_key(x, p, keep_path=True))
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                if len(xs) == 0:
                    return NoResult
                x = deep_merge_dicts(*xs)
            else:
                try:
                    x = search.get_pathlike_key(x, path, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    return NoResult
        if source is not None:
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            new_d = source.substitute(_template_context, eval_id="source")
            if checks.is_function(new_d):
                new_d = new_d(x)
        else:
            new_d = x
        return new_d


class SetAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.set`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "set"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        value: tp.Any,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if checks.is_function(value):
            if checks.is_builtin_func(value):
                value = RepFunc(lambda _value=value: _value)
            else:
                value = RepFunc(value)
        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                value=value,
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        value: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p[:-1])
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": x,
                    **(x if isinstance(x, dict) else {}),
                },
                template_context,
            )
            v = value.substitute(_template_context, eval_id="value", **kwargs)
            if checks.is_function(v):
                v = v(x)
            d = search.set_pathlike_key(d, p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class RemoveAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.remove`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "remove"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if isinstance(path, list):
            paths = [search.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search.resolve_pathlike_key(path)]
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            try:
                d = search.remove_pathlike_key(d, p, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class MoveAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.move`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "move"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Union[tp.PathMoveDict, tp.MaybeList[tp.PathLikeKey]],
        new_path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if new_path is None:
            checks.assert_instance_of(path, dict, arg_name="path")
            new_path = list(path.values())
            path = list(path.keys())
        if isinstance(path, list):
            paths = [search.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search.resolve_pathlike_key(path)]
        if isinstance(new_path, list):
            new_paths = [search.resolve_pathlike_key(p) for p in new_path]
        else:
            new_paths = [search.resolve_pathlike_key(new_path)]
        if len(paths) != len(new_paths):
            raise ValueError("Number of new paths must match number of paths")
        return (), {
            **dict(
                paths=paths,
                new_paths=new_paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        new_paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
    ) -> tp.Any:
        prev_keys = []
        for i, p in enumerate(paths):
            try:
                x = search.get_pathlike_key(d, p)
                d = search.remove_pathlike_key(d, p, make_copy=make_copy, prev_keys=prev_keys)
                d = search.set_pathlike_key(d, new_paths[i], x, make_copy=make_copy, prev_keys=prev_keys)
            except (KeyError, IndexError, AttributeError) as e:
                if not skip_missing:
                    raise e
                continue
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class RenameAssetFunc(MoveAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.rename`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "rename"

    @classmethod
    def prepare(
        cls,
        path: tp.Union[tp.PathRenameDict, tp.MaybeList[tp.PathLikeKey]],
        new_token: tp.Optional[tp.MaybeList[tp.PathKeyToken]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if new_token is None:
            checks.assert_instance_of(path, dict, arg_name="path")
            new_token = list(path.values())
            path = list(path.keys())
        if isinstance(path, list):
            paths = [search.resolve_pathlike_key(p) for p in path]
        else:
            paths = [search.resolve_pathlike_key(path)]
        if isinstance(new_token, list):
            new_tokens = [search.resolve_pathlike_key(t) for t in new_token]
        else:
            new_tokens = [search.resolve_pathlike_key(new_token)]
        if len(paths) != len(new_tokens):
            raise ValueError("Number of new tokens must match number of paths")
        new_paths = []
        for i in range(len(paths)):
            if len(new_tokens[i]) != 1:
                raise ValueError("Exactly one token must be provided for each path")
            new_paths.append(paths[i][:-1] + new_tokens[i])
        return (), {
            **dict(
                paths=paths,
                new_paths=new_paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }


class ReorderAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.reorder`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "reorder"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        new_order: tp.Union[str, tp.PathKeyTokens],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)

        if isinstance(new_order, str):
            if new_order.lower() in ("asc", "ascending"):
                new_order = lambda x: (
                    sorted(x)
                    if isinstance(x, dict)
                    else sorted(
                        range(len(x)),
                        key=x.__getitem__,
                    )
                )
            elif new_order.lower() in ("desc", "descending"):
                new_order = lambda x: (
                    sorted(x)
                    if isinstance(x, dict)
                    else sorted(
                        range(len(x)),
                        key=x.__getitem__,
                        reverse=True,
                    )
                )
        if isinstance(new_order, str):
            new_order = RepEval(new_order)
        elif checks.is_function(new_order):
            if checks.is_builtin_func(new_order):
                new_order = RepFunc(lambda _new_order=new_order: _new_order)
            else:
                new_order = RepFunc(new_order)
        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                new_order=new_order,
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                template_context=template_context,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        new_order: tp.Union[tp.PathKeyTokens, tp.CustomTemplate],
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            if isinstance(new_order, CustomTemplate):
                _template_context = flat_merge_dicts(
                    {
                        "d": d,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _new_order = new_order.substitute(_template_context, eval_id="new_order", **kwargs)
                if checks.is_function(_new_order):
                    _new_order = _new_order(x)
            else:
                _new_order = new_order
            if isinstance(x, dict):
                x = reorder_dict(x, _new_order, skip_missing=skip_missing)
            else:
                if checks.is_namedtuple(x):
                    x = type(x)(*reorder_list(x, _new_order, skip_missing=skip_missing))
                else:
                    x = type(x)(reorder_list(x, _new_order, skip_missing=skip_missing))
            d = search.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class QueryAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.query`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "query"

    _wrap: tp.ClassVar[tp.Optional[str]] = False

    @classmethod
    def prepare(
        cls,
        expression: tp.Union[str, tp.Callable, tp.CustomTemplate],
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)
        return_type = asset.resolve_setting(return_type, "return_type")

        if isinstance(expression, str):
            expression = RepEval(expression)
        elif checks.is_function(expression):
            if checks.is_builtin_func(expression):
                expression = RepFunc(lambda _expression=expression: _expression)
            else:
                expression = RepFunc(expression)
        elif not isinstance(expression, CustomTemplate):
            raise TypeError(f"Expression must be a string, function, or template")
        return (), {
            **dict(
                expression=expression,
                template_context=template_context,
                return_type=return_type,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        expression: tp.CustomTemplate,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        **kwargs,
    ) -> tp.Any:
        _template_context = flat_merge_dicts(
            {
                "d": d,
                "x": d,
                **search.search_config,
                **(d if isinstance(d, dict) else {}),
            },
            template_context,
        )
        new_d = expression.substitute(_template_context, eval_id="expression", **kwargs)
        if checks.is_function(new_d):
            new_d = new_d(d)
        if return_type is None:
            as_filter = True
        elif return_type.lower() == "bool":
            as_filter = False
        else:
            raise ValueError(f"Invalid return type: '{return_type}'")
        if as_filter and isinstance(new_d, bool):
            if new_d:
                return d
            return NoResult
        return new_d


class FindAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.find`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "find"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
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
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        per_path = asset.resolve_setting(per_path, "per_path")
        find_all = asset.resolve_setting(find_all, "find_all")
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        in_dumps = asset.resolve_setting(in_dumps, "in_dumps")
        dump_kwargs = asset.resolve_setting(dump_kwargs, "dump_kwargs", merge=True)
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)
        return_type = asset.resolve_setting(return_type, "return_type")
        return_path = asset.resolve_setting(return_path, "return_path")

        if path is not None:
            if isinstance(path, list):
                path = [search.resolve_pathlike_key(p) for p in path]
            else:
                path = search.resolve_pathlike_key(path)
        if per_path:
            if not isinstance(target, list):
                target = [target]
                if isinstance(path, list):
                    target *= len(path)
            if not isinstance(path, list):
                path = [path]
                if isinstance(target, list):
                    path *= len(target)
            if len(target) != len(path):
                raise ValueError("Number of targets must match number of paths")
        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(f"Source must be a string, function, or template")
        dump_kwargs = DumpAssetFunc.resolve_dump_kwargs(**dump_kwargs)
        contains_arg_names = set(get_func_arg_names(search.contains_in_obj))
        search_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in contains_arg_names}
        if "excl_types" not in search_kwargs:
            search_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                path=path,
                per_path=per_path,
                find_all=find_all,
                keep_path=keep_path,
                skip_missing=skip_missing,
                source=source,
                in_dumps=in_dumps,
                dump_kwargs=dump_kwargs,
                search_kwargs=search_kwargs,
                template_context=template_context,
                return_type=return_type,
                return_path=return_path,
            ),
            **kwargs,
        }

    @classmethod
    def match_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        find_all: bool = False,
        **kwargs,
    ) -> bool:
        """Match function for `FindAssetFunc.call`.

        Uses `vectorbtpro.utils.search.find` with `return_type="bool"` for text,
        and equality checks for other types.

        Target can be a function taking the value and returning a boolean. Target can also be an
        instance of `vectorbtpro.utils.search.Not` for negation."""
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        for target in targets:
            if isinstance(target, search.Not):
                target = target.value
                negation = True
            else:
                negation = False
            if checks.is_function(target):
                if target(d):
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif d is target:
                if (negation and find_all) or (not negation and not find_all):
                    return not negation
                continue
            elif d is None and target is None:
                if (negation and find_all) or (not negation and not find_all):
                    return not negation
                continue
            elif checks.is_bool(d) and checks.is_bool(target):
                if d == target:
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif checks.is_number(d) and checks.is_number(target):
                if d == target:
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif isinstance(d, str) and isinstance(target, str):
                if search.find(target, d, return_type="bool", **kwargs):
                    if (negation and find_all) or (not negation and not find_all):
                        return not negation
                    continue
            elif type(d) is type(target):
                try:
                    if d == target:
                        if (negation and find_all) or (not negation and not find_all):
                            return not negation
                        continue
                except Exception:
                    pass
            if (negation and not find_all) or (not negation and find_all):
                return negation
        if find_all:
            return True
        return False

    @classmethod
    def call(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: bool = True,
        find_all: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        source: tp.Optional[tp.CustomTemplate] = None,
        in_dumps: bool = False,
        dump_kwargs: tp.KwargsLike = None,
        search_kwargs: tp.KwargsLike = None,
        template_context: tp.KwargsLike = None,
        return_type: tp.Optional[str] = None,
        return_path: bool = False,
        **kwargs,
    ) -> tp.Any:
        if dump_kwargs is None:
            dump_kwargs = {}
        if search_kwargs is None:
            search_kwargs = {}
        if per_path:
            new_path_dct = {}
            new_list = []
            for i, p in enumerate(path):
                x = d
                try:
                    x = search.get_pathlike_key(x, p, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
                if source is not None:
                    _template_context = flat_merge_dicts(
                        {
                            "d": d,
                            "x": x,
                            **(x if isinstance(x, dict) else {}),
                        },
                        template_context,
                    )
                    _x = source.substitute(_template_context, eval_id="source")
                    if checks.is_function(_x):
                        x = _x(x)
                    else:
                        x = _x
                if not isinstance(x, str) and in_dumps:
                    x = dump(x, **dump_kwargs)
                t = target[i]
                if return_type is None or return_type.lower() == "bool":
                    if isinstance(t, search.Not):
                        t = t.value
                        negation = True
                    else:
                        negation = False
                    if search.contains_in_obj(
                        x,
                        cls.match_func,
                        target=t,
                        find_all=find_all,
                        **search_kwargs,
                        **kwargs,
                    ):
                        if negation:
                            if find_all:
                                return NoResult if return_type is None else False
                            continue
                        else:
                            if not find_all:
                                return d if return_type is None else True
                            continue
                    else:
                        if negation:
                            if not find_all:
                                return d if return_type is None else True
                            continue
                        else:
                            if find_all:
                                return NoResult if return_type is None else False
                            continue
                else:
                    path_dct = search.find_in_obj(
                        x,
                        cls.match_func,
                        target=t,
                        find_all=find_all,
                        **search_kwargs,
                        **kwargs,
                    )
                    if len(path_dct) == 0:
                        if find_all:
                            return {} if return_path else []
                        continue
                    if isinstance(t, search.Not):
                        raise TypeError("Target cannot be negated here")
                    if not isinstance(t, str):
                        raise ValueError("Target must be string")
                    for k, v in path_dct.items():
                        if not isinstance(v, str):
                            raise ValueError("Matched value must be string")
                        _return_type = "bool" if return_type.lower() == "field" else return_type
                        matches = search.find(t, v, return_type=_return_type, **kwargs)
                        if return_path:
                            if k not in new_path_dct:
                                new_path_dct[k] = []
                            if return_type.lower() == "field":
                                if matches:
                                    new_path_dct[k].append(v)
                            else:
                                new_path_dct[k].extend(matches)
                        else:
                            if return_type.lower() == "field":
                                if matches:
                                    new_list.append(v)
                            else:
                                new_list.extend(matches)
            if return_type is None or return_type.lower() == "bool":
                if find_all:
                    return d if return_type is None else True
                return NoResult if return_type is None else False
            else:
                if return_path:
                    return new_path_dct
                return new_list
        else:
            x = d
            if path is not None:
                if isinstance(path, list):
                    xs = []
                    for p in path:
                        try:
                            xs.append(search.get_pathlike_key(x, p, keep_path=True))
                        except (KeyError, IndexError, AttributeError) as e:
                            if not skip_missing:
                                raise e
                            continue
                    if len(xs) == 0:
                        if return_type is None:
                            return NoResult
                        if return_type.lower() == "bool":
                            return False
                        return {} if return_path else []
                    x = deep_merge_dicts(*xs)
                else:
                    try:
                        x = search.get_pathlike_key(x, path, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        if return_type is None:
                            return NoResult
                        if return_type.lower() == "bool":
                            return False
                        return {} if return_path else []
            if source is not None:
                _template_context = flat_merge_dicts(
                    {
                        "d": d,
                        "x": x,
                        **(x if isinstance(x, dict) else {}),
                    },
                    template_context,
                )
                _x = source.substitute(_template_context, eval_id="source")
                if checks.is_function(_x):
                    x = _x(x)
                else:
                    x = _x
            if not isinstance(x, str) and in_dumps:
                x = dump(x, **dump_kwargs)
            if return_type is None:
                if search.contains_in_obj(
                    x,
                    cls.match_func,
                    target=target,
                    find_all=find_all,
                    **search_kwargs,
                    **kwargs,
                ):
                    return d
                return NoResult
            elif return_type.lower() == "bool":
                return search.contains_in_obj(
                    x,
                    cls.match_func,
                    target=target,
                    find_all=find_all,
                    **search_kwargs,
                    **kwargs,
                )
            else:
                path_dct = search.find_in_obj(
                    x,
                    cls.match_func,
                    target=target,
                    find_all=find_all,
                    **search_kwargs,
                    **kwargs,
                )
                if len(path_dct) == 0:
                    return {} if return_path else []
                if not isinstance(target, list):
                    targets = [target]
                else:
                    targets = target
                new_path_dct = {}
                new_list = []
                for target in targets:
                    if isinstance(target, search.Not):
                        raise TypeError("Target cannot be negated here")
                    if not isinstance(target, str):
                        raise ValueError("Target must be string")
                    for k, v in path_dct.items():
                        if not isinstance(v, str):
                            raise ValueError("Matched value must be string")
                        _return_type = "bool" if return_type.lower() == "field" else return_type
                        matches = search.find(target, v, return_type=_return_type, **kwargs)
                        if return_path:
                            if k not in new_path_dct:
                                new_path_dct[k] = []
                            if return_type.lower() == "field":
                                if matches:
                                    new_path_dct[k].append(v)
                            else:
                                new_path_dct[k].extend(matches)
                        else:
                            if return_type.lower() == "field":
                                if matches:
                                    new_list.append(v)
                            else:
                                new_list.extend(matches)
                if return_path:
                    return new_path_dct
                return new_list


class FindReplaceAssetFunc(FindAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.find_replace`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "find_replace"

    @classmethod
    def prepare(
        cls,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        replacement: tp.Optional[tp.MaybeList[tp.Any]] = None,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        per_path = asset.resolve_setting(per_path, "per_path")
        find_all = asset.resolve_setting(find_all, "find_all")
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if replacement is None:
            checks.assert_instance_of(target, dict, arg_name="path")
            replacement = list(target.values())
            target = list(target.keys())
        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
                if isinstance(target, list):
                    paths *= len(target)
                elif isinstance(replacement, list):
                    paths *= len(replacement)
        else:
            paths = [None]
            if isinstance(target, list):
                paths *= len(target)
            elif isinstance(replacement, list):
                paths *= len(replacement)
        if per_path:
            if not isinstance(target, list):
                target = [target] * len(paths)
            if not isinstance(replacement, list):
                replacement = [replacement] * len(paths)
            if len(target) != len(replacement) != len(paths):
                raise ValueError("Number of targets and replacements must match number of paths")
        find_arg_names = set(get_func_arg_names(search.find_in_obj))
        find_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in find_arg_names}
        if "excl_types" not in find_kwargs:
            find_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                replacement=replacement,
                paths=paths,
                per_path=per_path,
                find_all=find_all,
                keep_path=keep_path,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                find_kwargs=find_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def replace_func(
        cls,
        k: tp.Optional[tp.Hashable],
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        **kwargs,
    ) -> tp.Any:
        """Replace function for `FindReplaceAssetFunc.call`.

        Uses `vectorbtpro.utils.search.replace` for text and returns replacement for other types.

        Target can be a function taking the value and returning a boolean.
        Replacement can also be a function taking the value and returning a new value."""
        if not isinstance(target, list):
            targets = [target]
        else:
            targets = target
        if not isinstance(replacement, list):
            replacements = [replacement]
            if len(targets) > 1 and len(replacements) == 1:
                replacements *= len(targets)
        else:
            replacements = replacement
        if len(targets) != len(replacements):
            raise ValueError("Number of targets must match number of replacements")
        for i, target in enumerate(targets):
            if isinstance(target, search.Not):
                raise TypeError("Target cannot be negated here")
            replacement = replacements[i]
            if checks.is_function(replacement):
                replacement = replacement(d)
            if checks.is_function(target):
                if target(d):
                    return replacement
            elif d is target:
                return replacement
            elif d is None and target is None:
                return replacement
            elif checks.is_bool(d) and checks.is_bool(target):
                if d == target:
                    return replacement
            elif checks.is_number(d) and checks.is_number(target):
                if d == target:
                    return replacement
            elif isinstance(d, str) and isinstance(target, str):
                d = search.replace(target, replacement, d, **kwargs)
            elif type(d) is type(target):
                try:
                    if d == target:
                        return replacement
                except Exception:
                    pass
        return d

    @classmethod
    def call(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        replacement: tp.MaybeList[tp.Any],
        paths: tp.List[tp.PathLikeKey],
        per_path: bool = True,
        find_all: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        find_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if find_kwargs is None:
            find_kwargs = {}
        prev_keys = []
        found_all = True
        if find_all:
            for i, p in enumerate(paths):
                x = d
                if p is not None:
                    try:
                        x = search.get_pathlike_key(x, p, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                path_dct = search.find_in_obj(
                    x,
                    cls.match_func,
                    target=target[i] if per_path else target,
                    find_all=find_all,
                    **find_kwargs,
                    **kwargs,
                )
                if len(path_dct) == 0:
                    found_all = False
                    break
        if found_all:
            for i, p in enumerate(paths):
                x = d
                if p is not None:
                    try:
                        x = search.get_pathlike_key(x, p, keep_path=keep_path)
                    except (KeyError, IndexError, AttributeError) as e:
                        if not skip_missing:
                            raise e
                        continue
                path_dct = search.find_in_obj(
                    x,
                    cls.match_func,
                    target=target[i] if per_path else target,
                    find_all=find_all,
                    **find_kwargs,
                    **kwargs,
                )
                for k, v in path_dct.items():
                    if p is not None and not keep_path:
                        new_p = search.combine_pathlike_keys(p, k, minimize=True)
                    else:
                        new_p = k
                    v = cls.replace_func(
                        k,
                        v,
                        target[i] if per_path else target,
                        replacement[i] if per_path else replacement,
                        **kwargs,
                    )
                    d = search.set_pathlike_key(d, new_p, v, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class FindRemoveAssetFunc(FindAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.find_remove`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "find_remove"

    @classmethod
    def prepare(
        cls,
        target: tp.Union[dict, tp.MaybeList[tp.Any]],
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        per_path: tp.Optional[bool] = None,
        find_all: tp.Optional[bool] = None,
        keep_path: tp.Optional[bool] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        per_path = asset.resolve_setting(per_path, "per_path")
        find_all = asset.resolve_setting(find_all, "find_all")
        keep_path = asset.resolve_setting(keep_path, "keep_path")
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
                if isinstance(target, list):
                    paths *= len(target)
        else:
            paths = [None]
            if isinstance(target, list):
                paths *= len(target)
        if per_path:
            if not isinstance(target, list):
                target = [target] * len(paths)
            if len(target) != len(paths):
                raise ValueError("Number of targets must match number of paths")
        find_arg_names = set(get_func_arg_names(search.find_in_obj))
        find_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in find_arg_names}
        if "excl_types" not in find_kwargs:
            find_kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                target=target,
                paths=paths,
                per_path=per_path,
                find_all=find_all,
                keep_path=keep_path,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
                find_kwargs=find_kwargs,
            ),
            **kwargs,
        }

    @classmethod
    def is_empty_func(cls, d: tp.Any) -> bool:
        """Return whether object is empty."""
        if d is None:
            return True
        if checks.is_collection(d) and len(d) == 0:
            return True
        return False

    @classmethod
    def call(
        cls,
        d: tp.Any,
        target: tp.MaybeList[tp.Any],
        paths: tp.List[tp.PathLikeKey],
        per_path: bool = True,
        find_all: bool = False,
        keep_path: bool = False,
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        find_kwargs: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if find_kwargs is None:
            find_kwargs = {}
        prev_keys = []
        new_p_v_map = {}
        for i, p in enumerate(paths):
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p, keep_path=keep_path)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            path_dct = search.find_in_obj(
                x,
                cls.match_func,
                target=target[i] if per_path else target,
                find_all=find_all,
                **find_kwargs,
                **kwargs,
            )
            if len(path_dct) == 0 and find_all:
                new_p_v_map = {}
                break
            for k, v in path_dct.items():
                if p is not None and not keep_path:
                    new_p = search.combine_pathlike_keys(p, k, minimize=True)
                else:
                    new_p = k
                new_p_v_map[new_p] = v
        for new_p, v in new_p_v_map.items():
            d = search.remove_pathlike_key(d, new_p, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class FlattenAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.flatten`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "flatten"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        if "excl_types" not in kwargs:
            kwargs["excl_types"] = (tuple, set, frozenset)
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search.flatten_obj(x, **kwargs)
            d = search.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class UnflattenAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.unflatten`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "unflatten"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def prepare(
        cls,
        path: tp.Optional[tp.MaybeList[tp.PathLikeKey]] = None,
        skip_missing: tp.Optional[bool] = None,
        make_copy: tp.Optional[bool] = None,
        changed_only: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        skip_missing = asset.resolve_setting(skip_missing, "skip_missing")
        make_copy = asset.resolve_setting(make_copy, "make_copy")
        changed_only = asset.resolve_setting(changed_only, "changed_only")

        if path is not None:
            if isinstance(path, list):
                paths = [search.resolve_pathlike_key(p) for p in path]
            else:
                paths = [search.resolve_pathlike_key(path)]
        else:
            paths = [None]
        return (), {
            **dict(
                paths=paths,
                skip_missing=skip_missing,
                make_copy=make_copy,
                changed_only=changed_only,
            ),
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        paths: tp.List[tp.PathLikeKey],
        skip_missing: bool = False,
        make_copy: bool = True,
        changed_only: bool = False,
        **kwargs,
    ) -> tp.Any:
        prev_keys = []
        for p in paths:
            x = d
            if p is not None:
                try:
                    x = search.get_pathlike_key(x, p)
                except (KeyError, IndexError, AttributeError) as e:
                    if not skip_missing:
                        raise e
                    continue
            x = search.unflatten_obj(x, **kwargs)
            d = search.set_pathlike_key(d, p, x, make_copy=make_copy, prev_keys=prev_keys)
        if not changed_only or len(prev_keys) > 0:
            return d
        return NoResult


class DumpAssetFunc(AssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.dump`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "dump"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    @classmethod
    def resolve_dump_kwargs(
        cls,
        dump_engine: tp.Optional[str] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.Kwargs:
        """Resolve keyword arguments related to dumping."""
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        dump_engine = asset.resolve_setting(dump_engine, "dump_engine")
        kwargs = asset.resolve_setting(kwargs, f"dump_engine_kwargs.{dump_engine}", default={}, merge=True)
        return {"dump_engine": dump_engine, **kwargs}

    @classmethod
    def prepare(
        cls,
        source: tp.Union[None, str, tp.Callable, tp.CustomTemplate] = None,
        dump_engine: tp.Optional[str] = None,
        template_context: tp.KwargsLike = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        template_context = asset.resolve_setting(template_context, "template_context", merge=True)
        dump_kwargs = cls.resolve_dump_kwargs(dump_engine=dump_engine, **kwargs)

        if source is not None:
            if isinstance(source, str):
                source = RepEval(source)
            elif checks.is_function(source):
                if checks.is_builtin_func(source):
                    source = RepFunc(lambda _source=source: _source)
                else:
                    source = RepFunc(source)
            elif not isinstance(source, CustomTemplate):
                raise TypeError(f"Source must be a string, function, or template")
        return (), {
            **dict(
                source=source,
                template_context=template_context,
            ),
            **dump_kwargs,
            **kwargs,
        }

    @classmethod
    def call(
        cls,
        d: tp.Any,
        source: tp.Optional[CustomTemplate] = None,
        dump_engine: str = "nestedtext",
        template_context: tp.KwargsLike = None,
        **kwargs,
    ) -> tp.Any:
        if source is not None:
            _template_context = flat_merge_dicts(
                {
                    "d": d,
                    "x": d,
                    **(d if isinstance(d, dict) else {}),
                },
                template_context,
            )
            new_d = source.substitute(_template_context, eval_id="source")
            if checks.is_function(new_d):
                new_d = new_d(d)
        else:
            new_d = d
        return dump(new_d, dump_engine=dump_engine, **kwargs)


# ############# Reduce classes ############# #


class ReduceAssetFunc(AssetFunc):
    """Abstract asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.reduce`."""

    _wrap: tp.ClassVar[tp.Optional[str]] = False

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = None

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any, *args, **kwargs) -> tp.Any:
        raise NotImplementedError

    @classmethod
    def prepare_and_call(cls, d1: tp.Any, d2: tp.Any, *args, **kwargs):
        args, kwargs = cls.prepare(*args, **kwargs)
        return cls.call(d1, d2, *args, **kwargs)


class CollectAssetFunc(ReduceAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.collect`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "collect"

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = {}

    @classmethod
    def prepare(
        cls,
        sort_keys: tp.Optional[bool] = None,
        asset: tp.Optional[tp.MaybeType[tp.KnowledgeAsset]] = None,
        **kwargs,
    ) -> tp.ArgsKwargs:
        if asset is None:
            from vectorbtpro.utils.knowledge.base_assets import KnowledgeAsset

            asset = KnowledgeAsset
        sort_keys = asset.resolve_setting(sort_keys, "sort_keys")

        return (), {**dict(sort_keys=sort_keys), **kwargs}

    @classmethod
    def sort_key(cls, k: tp.Any) -> tuple:
        """Function for sorting keys."""
        return (0, k) if isinstance(k, str) else (1, k)

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any, sort_keys: bool = False) -> tp.Any:
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            raise TypeError("Data items must be dicts")
        new_d1 = dict(d1)
        for k1 in d1:
            if k1 not in new_d1:
                new_d1[k1] = [d1[k1]]
            if k1 in d2:
                new_d1[k1].append(d2[k1])
        for k2 in d2:
            if k2 not in new_d1:
                new_d1[k2] = [d2[k2]]
        if sort_keys:
            return dict(sorted(new_d1.items(), key=lambda x: cls.sort_key(x[0])))
        return new_d1


class MergeDictsAssetFunc(ReduceAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.merge_dicts`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "merge_dicts"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = {}

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any, **kwargs) -> tp.Any:
        if not isinstance(d1, dict) or not isinstance(d2, dict):
            raise TypeError("Data items must be dicts")
        return merge_dicts(d1, d2, **kwargs)


class MergeListsAssetFunc(ReduceAssetFunc):
    """Asset function class for `vectorbtpro.utils.knowledge.base_assets.KnowledgeAsset.merge_lists`."""

    _short_name: tp.ClassVar[tp.Optional[str]] = "merge_lists"

    _wrap: tp.ClassVar[tp.Optional[str]] = True

    _initializer: tp.ClassVar[tp.Optional[tp.Any]] = []

    @classmethod
    def call(cls, d1: tp.Any, d2: tp.Any) -> tp.Any:
        if not isinstance(d1, list) or not isinstance(d2, list):
            raise TypeError("Data items must be lists")
        return d1 + d2
