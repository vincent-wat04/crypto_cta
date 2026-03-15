"""
Factor expression system.

Allows defining factors as string expressions that can be parsed and evaluated.
Supports 1st-order (single operator) and 2nd-order (nested operators) expressions.

Expression format: "operator(field, param)" or "operator1(operator2(field, p2), p1)"

Examples:
  "ts_rank(return_1, 60)"
  "delta(ts_mean(volume, 20), 5)"
  "ts_corr(return_1, trade_imbalance, 30)"
  "decay_linear(delta(return_1, 5), 20)"
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from .operators import UNARY_OPERATORS, BINARY_OPERATORS


class FactorExpression:
    """
    Parsed factor expression that can be evaluated on data.
    """

    def __init__(self, expr_str: str):
        self.expr_str = expr_str.strip()
        self._parsed = _parse_expression(self.expr_str)

    def evaluate(self, data: pd.DataFrame) -> pd.Series:
        """Evaluate this expression on a DataFrame of primitives."""
        return _eval_node(self._parsed, data)

    @property
    def order(self) -> int:
        """Number of TS operator applications (depth of operator nesting)."""
        return _count_operators(self._parsed)

    @property
    def fields_used(self) -> List[str]:
        """List of primitive fields referenced."""
        return _collect_fields(self._parsed)

    def __repr__(self):
        return f"FactorExpression('{self.expr_str}')"


# ══════════════════════════════════════════════════════════
# AST Nodes
# ══════════════════════════════════════════════════════════

class _FieldNode:
    """Leaf node: references a column in the data."""
    def __init__(self, name: str):
        self.name = name


class _ConstNode:
    """Leaf node: numeric constant."""
    def __init__(self, value: float):
        self.value = value


class _UnaryOpNode:
    """Unary operator: op(child, param)."""
    def __init__(self, op_name: str, child, param):
        self.op_name = op_name
        self.child = child
        self.param = param  # int or float


class _BinaryOpNode:
    """Binary operator: op(child1, child2, param)."""
    def __init__(self, op_name: str, child1, child2, param):
        self.op_name = op_name
        self.child1 = child1
        self.child2 = child2
        self.param = param


# ══════════════════════════════════════════════════════════
# Parser
# ══════════════════════════════════════════════════════════

def _parse_expression(expr: str):
    """Parse expression string into AST."""
    expr = expr.strip()

    # Try to match function call: name(args)
    match = re.match(r'^(\w+)\((.+)\)$', expr)
    if match:
        func_name = match.group(1)
        args_str = match.group(2)
        args = _split_args(args_str)

        # Unary operators (1 data arg + 1 param)
        if func_name in UNARY_OPERATORS:
            if func_name in ("signedpower",):
                # signedpower(x, a) where a is float
                child = _parse_expression(args[0])
                param = float(args[1])
                return _UnaryOpNode(func_name, child, param)
            elif func_name in ("log1p_safe", "abs_val"):
                child = _parse_expression(args[0])
                return _UnaryOpNode(func_name, child, None)
            elif func_name == "decay_exp" and len(args) == 3:
                child = _parse_expression(args[0])
                return _UnaryOpNode(func_name, child, (int(args[1]), float(args[2])))
            else:
                child = _parse_expression(args[0])
                param = int(args[1]) if len(args) > 1 else None
                return _UnaryOpNode(func_name, child, param)

        # Binary operators (2 data args + 1 param)
        if func_name in BINARY_OPERATORS:
            child1 = _parse_expression(args[0])
            child2 = _parse_expression(args[1])
            param = int(args[2]) if len(args) > 2 else 20
            return _BinaryOpNode(func_name, child1, child2, param)

    # Numeric constant
    try:
        return _ConstNode(float(expr))
    except ValueError:
        pass

    # Field name
    return _FieldNode(expr)


def _split_args(args_str: str) -> List[str]:
    """Split function arguments respecting nested parentheses."""
    args = []
    depth = 0
    current = []
    for ch in args_str:
        if ch == '(':
            depth += 1
            current.append(ch)
        elif ch == ')':
            depth -= 1
            current.append(ch)
        elif ch == ',' and depth == 0:
            args.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append(''.join(current).strip())
    return args


# ══════════════════════════════════════════════════════════
# Evaluator
# ══════════════════════════════════════════════════════════

def _eval_node(node, data: pd.DataFrame) -> pd.Series:
    """Recursively evaluate AST node on data."""
    if isinstance(node, _FieldNode):
        if node.name not in data.columns:
            raise KeyError(f"Field '{node.name}' not found in data. Available: {list(data.columns[:20])}")
        return data[node.name].astype(float)

    if isinstance(node, _ConstNode):
        return pd.Series(node.value, index=data.index)

    if isinstance(node, _UnaryOpNode):
        child_series = _eval_node(node.child, data)
        op_func = UNARY_OPERATORS[node.op_name]

        if node.op_name in ("log1p_safe", "abs_val"):
            return op_func(child_series)
        elif node.op_name == "signedpower":
            return op_func(child_series, node.param)
        elif node.op_name == "decay_exp" and isinstance(node.param, tuple):
            return op_func(child_series, node.param[0], node.param[1])
        else:
            return op_func(child_series, node.param)

    if isinstance(node, _BinaryOpNode):
        child1_series = _eval_node(node.child1, data)
        child2_series = _eval_node(node.child2, data)
        op_func = BINARY_OPERATORS[node.op_name]
        return op_func(child1_series, child2_series, node.param)

    raise ValueError(f"Unknown node type: {type(node)}")


def _count_operators(node) -> int:
    """Count TS operator depth."""
    if isinstance(node, (_FieldNode, _ConstNode)):
        return 0
    if isinstance(node, _UnaryOpNode):
        return 1 + _count_operators(node.child)
    if isinstance(node, _BinaryOpNode):
        return 1 + max(_count_operators(node.child1), _count_operators(node.child2))
    return 0


def _collect_fields(node) -> List[str]:
    """Collect all field names referenced."""
    if isinstance(node, _FieldNode):
        return [node.name]
    if isinstance(node, _ConstNode):
        return []
    if isinstance(node, _UnaryOpNode):
        return _collect_fields(node.child)
    if isinstance(node, _BinaryOpNode):
        return _collect_fields(node.child1) + _collect_fields(node.child2)
    return []
