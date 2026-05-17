# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for chaining."""

import inspect

from vectorbtpro import _typing as tp
from vectorbtpro.utils.decorators import hybrid_method

__all__ = [
    "Chainable",
]


class Chainable:
    """Class representing an object that can be chained."""

    @hybrid_method
    def pipe(cls_or_self, func: tp.PipeFunc, *args, **kwargs) -> tp.Any:
        """Apply a chainable function that expects a `Chainable` instance.

        Can be called as a class method, but then will pass only `*args` and `**kwargs`.

        Argument `func` can be a function, a string denoting a (deep) attribute to be resolved
        with `vectorbtpro.utils.attr_.deep_getattr`, or a tuple where the first element is one
        of the above and the second element is a positional argument or keyword argument where
        to pass the instance. If not a tuple, passes the instance as the first positional argument.
        If a string and the target function is an instance method, won't pass the instance since
        it's already bound to this instance."""
        if isinstance(func, tuple):
            func, arg_name = func
            if not isinstance(cls_or_self, type):
                if isinstance(arg_name, int):
                    args = list(args)
                    args.insert(arg_name, cls_or_self)
                    args = tuple(args)
                else:
                    kwargs[arg_name] = cls_or_self
            prepend_to_args = False
        else:
            prepend_to_args = not isinstance(cls_or_self, type)
        if isinstance(func, str):
            from vectorbtpro.utils.attr_ import deep_getattr

            func = deep_getattr(cls_or_self, func, call_last_attr=False)
            if not callable(func) and len(args) == 0 and len(kwargs) == 0:
                return func
            if prepend_to_args:
                prepend_to_args = not inspect.ismethod(func)
        if prepend_to_args:
            args = (cls_or_self, *args)
        return func(*args, **kwargs)

    @hybrid_method
    def chain(cls_or_self, tasks: tp.PipeTasks) -> tp.Any:
        """Chain multiple tasks with `Chainable.pipe`."""
        from vectorbtpro.utils.execution import Task

        result = cls_or_self
        for task in tasks:
            if not isinstance(task, Task):
                if isinstance(task, tuple):
                    task = Task.from_tuple(task)
                else:
                    task = Task(task)
            func, args, kwargs = task
            result = result.pipe(func, *args, **kwargs)
        return result
