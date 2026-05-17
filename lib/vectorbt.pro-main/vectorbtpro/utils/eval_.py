# Copyright (c) 2021-2024 Oleg Polakow. All rights reserved.

"""Utilities for evaluation and compilation."""

import ast
import inspect
import symtable
import builtins

from vectorbtpro import _typing as tp
from vectorbtpro.utils import checks

__all__ = [
    "evaluate",
]


def evaluate(expr: str, context: tp.KwargsLike = None) -> tp.Any:
    """Evaluate one to multiple lines of expression.

    Returns the result of the last line."""
    expr = inspect.cleandoc(expr)
    if context is None:
        context = {}
    if "\n" in expr:
        tree = ast.parse(expr)
        eval_expr = ast.Expression(tree.body[-1].value)
        exec_expr = ast.Module(tree.body[:-1], type_ignores=[])
        exec(compile(exec_expr, "<string>", "exec"), context)
        return eval(compile(eval_expr, "<string>", "eval"), context)
    return eval(compile(expr, "<string>", "eval"), context)


def get_symbols(table: symtable.SymbolTable) -> tp.List[symtable.Symbol]:
    """Get symbols from a symbol table recursively."""
    symbols = []
    children = {child.get_name(): child for child in table.get_children()}
    for symbol in table.get_symbols():
        if symbol.is_namespace():
            symbols.extend(get_symbols(children[symbol.get_name()]))
        else:
            symbols.append(symbol)
    return symbols


def get_free_vars(expr: str) -> tp.List[str]:
    """Parse the code and retrieve all free variables, excluding built-in names."""
    expr = inspect.cleandoc(expr)
    global_table = symtable.symtable(expr, "<string>", "exec")
    symbols = get_symbols(global_table)
    builtins_set = set(dir(builtins))
    free_vars = []
    free_vars_set = set()
    not_free_vars_set = set()
    for symbol in symbols:
        symbol_name = symbol.get_name()
        if symbol.is_imported() or symbol.is_parameter() or symbol.is_assigned() or symbol_name in builtins_set:
            not_free_vars_set.add(symbol_name)
    for symbol in symbols:
        symbol_name = symbol.get_name()
        if symbol_name not in not_free_vars_set and symbol_name not in free_vars_set:
            free_vars.append(symbol_name)
            free_vars_set.add(symbol_name)
    return free_vars


class Evaluable:
    """Abstract class for instances that can be evaluated."""

    def meets_eval_id(self, eval_id: tp.Optional[tp.Hashable]) -> bool:
        """Return whether the evaluation id of the instance meets the global evaluation id."""
        if self.eval_id is not None and eval_id is not None:
            if checks.is_complex_sequence(self.eval_id):
                if eval_id not in self.eval_id:
                    return False
            else:
                if eval_id != self.eval_id:
                    return False
        return True
