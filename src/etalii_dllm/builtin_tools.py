"""Built-in tools whose results are deterministic, so a whole agent run repeats bit for bit.

An agent run is only as reproducible as its tools. These need no external MCP server and return the same text for
the same arguments on every machine:

- ``calculator``: ``calculate(expression)``, exact rational arithmetic (``+ - * / // % **`` with integer exponents,
  parentheses, decimal literals); never a binary float.
- ``files=DIR``: ``list_files(path)`` (sorted by name) and ``read_file(path, start, lines)``, read-only and confined
  to ``DIR``.
- ``documents``: ``search_documents(query, top)`` over the engine's document index (``--index``).

They are served as an in-process MCP server (``dllm chat --tool calculator --tool files=. ...``) or over stdio
(``dllm-tools --tool calculator``), so an agent sees them like any other MCP tools.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError as McpToolError
from mcp.types import ToolAnnotations

TOOLS = ("calculator", "files", "documents")
DECIMALS = 30
"""Digits after the point in a calculator answer that does not terminate (truncated, marked with ``≈``)."""
MAX_EXPONENT = 1024
MAX_BITS = 1 << 14
"""Largest numerator or denominator (in bits) an intermediate result may have."""
MAX_LINES = 200
MAX_EXPRESSION = 1000

_DETERMINISTIC = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False)


class ToolError(ValueError):
    """A tool call that cannot be answered; the model sees the message."""


# -- calculator -----------------------------------------------------------------------------------------------------


def _check(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
        raise ToolError("the result is too large")
    return value


def evaluate(expression: str) -> Fraction:
    """The exact value of an arithmetic expression."""
    if len(expression) > MAX_EXPRESSION:
        raise ToolError(f"the expression is longer than {MAX_EXPRESSION} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        raise ToolError(f"not an arithmetic expression: {expression!r}") from None
    source = expression.strip()

    def value(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return Fraction(ast.get_source_segment(source, node) or str(node.value))  # the decimal as written
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.UAdd | ast.USub):
            operand = value(node.operand)
            return operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp):
            left, right = value(node.left), value(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return _check(left + right)
            if isinstance(op, ast.Sub):
                return _check(left - right)
            if isinstance(op, ast.Mult):
                return _check(left * right)
            if isinstance(op, ast.Div | ast.FloorDiv | ast.Mod):
                if right == 0:
                    raise ToolError("division by zero")
                if isinstance(op, ast.Div):
                    return _check(left / right)
                return _check(Fraction(left // right) if isinstance(op, ast.FloorDiv) else left % right)
            if isinstance(op, ast.Pow):
                if right.denominator != 1:
                    raise ToolError("only integer exponents are exact")
                if abs(right.numerator) > MAX_EXPONENT:
                    raise ToolError(f"exponents are limited to {MAX_EXPONENT}")
                if left == 0 and right < 0:
                    raise ToolError("division by zero")
                if max(left.numerator.bit_length(), left.denominator.bit_length()) * abs(right.numerator) > MAX_BITS:
                    raise ToolError("the result is too large")
                return _check(left**right.numerator)
        raise ToolError(f"unsupported in an arithmetic expression: {ast.unparse(node)!r}")

    return value(tree.body)


def decimal(value: Fraction, places: int = DECIMALS) -> str:
    """``value`` in decimal: exact when it terminates within ``places`` digits, else truncated with ``≈``."""
    sign = "-" if value < 0 else ""
    numerator, denominator = abs(value.numerator), value.denominator
    whole, remainder = divmod(numerator, denominator)
    digits = []
    while remainder and len(digits) < places:
        digit, remainder = divmod(remainder * 10, denominator)
        digits.append(str(digit))
    text = sign + str(whole) + ("." + "".join(digits) if digits else "")
    return ("≈ " if remainder else "") + text


def calculate(expression: str) -> str:
    value = evaluate(expression)
    exact = str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    shown = decimal(value)
    return exact if shown == exact else f"{exact} = {shown}" if not shown.startswith("≈") else f"{exact} {shown}"


# -- files ----------------------------------------------------------------------------------------------------------


class Files:
    """Read-only access to one directory tree."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise ToolError(f"{root}: not a directory")

    def _path(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if resolved != self.root and self.root not in resolved.parents:
            raise ToolError(f"{path!r} is outside the shared directory")
        return resolved

    def _name(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix() or "."

    def list_files(self, path: str = ".") -> str:
        directory = self._path(path)
        if not directory.is_dir():
            raise ToolError(f"{path!r} is not a directory")
        entries = sorted(directory.iterdir(), key=lambda p: p.name)
        lines = [self._name(p) + ("/" if p.is_dir() else f"  ({p.stat().st_size} bytes)") for p in entries]
        return "\n".join(lines) if lines else "(empty)"

    def read_file(self, path: str, start: int = 1, lines: int = MAX_LINES) -> str:
        file = self._path(path)
        if not file.is_file():
            raise ToolError(f"{path!r} is not a file")
        if start < 1 or lines < 1:
            raise ToolError("start and lines must be at least 1")
        text = file.read_bytes().decode("utf-8", errors="replace").splitlines()
        lines = min(lines, MAX_LINES)
        chosen = text[start - 1 : start - 1 + lines]
        end = start - 1 + len(chosen)
        header = f"{self._name(file)}: lines {start}-{end} of {len(text)}" if chosen else f"{self._name(file)}: "
        if not chosen:
            header += f"no lines from {start} (the file has {len(text)})"
        return "\n".join([header, *chosen])


# -- the server -----------------------------------------------------------------------------------------------------


def parse(specs: Sequence[str]) -> dict[str, str | None]:
    """``["calculator", "files=docs"]`` -> ``{"calculator": None, "files": "docs"}``, checked."""
    chosen: dict[str, str | None] = {}
    for spec in specs:
        name, _, argument = spec.partition("=")
        if name not in TOOLS:
            raise ToolError(f"unknown built-in tool {name!r}; available: {', '.join(TOOLS)}")
        if name == "files" and not argument:
            raise ToolError("files needs a directory: --tool files=DIR")
        chosen[name] = argument or None
    return chosen


def _answer(function: Any, *args: Any) -> str:
    """The tool's answer; a :class:`ToolError` reaches the model as an error result with its message."""
    try:
        return function(*args)
    except ToolError as error:
        raise McpToolError(str(error)) from None


def server(specs: Sequence[str], engine: Any = None) -> MCPServer:
    """An MCP server with the chosen built-in tools. ``documents`` needs ``engine`` (else the default engine)."""
    chosen = parse(specs)
    tools = MCPServer("dllm-tools")
    if "calculator" in chosen:

        @tools.tool(name="calculate", annotations=_DETERMINISTIC)
        def calculate_tool(expression: str) -> str:
            """Calculates an arithmetic expression exactly (+ - * / // % ** and parentheses; decimals are exact).

            Returns the exact value as a fraction and its decimal expansion."""
            return _answer(calculate, expression)

    if "files" in chosen:
        files = Files(chosen["files"] or ".")

        @tools.tool(name="list_files", annotations=_DETERMINISTIC)
        def list_files(path: str = ".") -> str:
            """Lists a directory of the shared files, sorted by name (directories end with /)."""
            return _answer(files.list_files, path)

        @tools.tool(name="read_file", annotations=_DETERMINISTIC)
        def read_file(path: str, start: int = 1, lines: int = MAX_LINES) -> str:
            """Reads lines of a text file from the shared files (at most 200 lines per call)."""
            return _answer(files.read_file, path, start, lines)

    if "documents" in chosen:

        @tools.tool(name="search_documents", annotations=_DETERMINISTIC)
        def search_documents(query: str, top: int = 5) -> str:
            """Finds the passages of the document index closest to the query, best first (JSON)."""
            from etalii_dllm.engine import default_engine

            retriever = (engine or default_engine()).retriever
            if retriever is None:
                raise McpToolError("no document index; start with --index (dllm index build)")
            hits = retriever.search(query, top)
            rows = [{"rank": h.rank, "score": h.score, **h.chunk.to_json()} for h in hits]
            return json.dumps(rows, ensure_ascii=False, indent=2)

    return tools


def main(argv: list[str] | None = None) -> None:
    from etalii_dllm.engine import add_runtime_arguments, use_model_file

    parser = argparse.ArgumentParser(prog="dllm-tools", description="EtAlii.Dllm built-in tools as an MCP server")
    parser.add_argument("--tool", action="append", default=[], metavar="NAME[=ARG]",
                        help=f"a built-in tool to offer (repeatable): {', '.join(TOOLS)}")  # fmt: skip
    add_runtime_arguments(parser)
    args = parser.parse_args(argv)
    if args.index or args.model:
        use_model_file(args.model, index=args.index, embedding_model=args.embedding_model)
    try:
        tools = server(args.tool or list(TOOLS[:1]))
    except ToolError as error:
        parser.error(str(error))
    tools.run("stdio")
