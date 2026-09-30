"""Renders a model's own Jinja chat template, the way Hugging Face ``apply_chat_template`` does.

The environment matches ``transformers``: a sandbox, ``trim_blocks`` and ``lstrip_blocks``, the ``loopcontrols``
extension, a ``tojson`` filter that keeps non-ASCII text, and ``raise_exception``. The template also defines how
tools are presented to the model, so the OpenAI and MCP tool support builds on it.

Determinism: templates only see the request. ``strftime_now`` (used by some templates for the current date) is
given the date the caller passes, never the clock; without one it is undefined, so templates that check for it (Llama
3.x) use their own fixed default date and templates that do not fail.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Mapping, Sequence
from typing import Any

import jinja2
from jinja2.ext import loopcontrols
from jinja2.sandbox import ImmutableSandboxedEnvironment


class ChatTemplateError(ValueError):
    """The template is invalid or raised an error (for example an unsupported role order)."""


def _raise(message: str) -> None:
    raise ChatTemplateError(message)


def _tojson(value: Any, ensure_ascii: bool = False, indent: int | None = None, separators: Any = None,
            sort_keys: bool = False) -> str:  # fmt: skip
    return json.dumps(value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)


class ChatTemplate:
    def __init__(self, source: str, *, special_tokens: Mapping[str, str | None] | None = None) -> None:
        """``special_tokens`` holds ``bos_token``, ``eos_token`` and friends, which templates reference."""
        environment = ImmutableSandboxedEnvironment(
            trim_blocks=True, lstrip_blocks=True, extensions=[loopcontrols], undefined=jinja2.Undefined
        )
        environment.filters["tojson"] = _tojson
        environment.globals["raise_exception"] = _raise
        try:
            self._template = environment.from_string(source)
        except jinja2.TemplateSyntaxError as error:
            raise ChatTemplateError(f"invalid chat template: {error}") from error
        self.source = source
        self._special_tokens = {k: v for k, v in (special_tokens or {}).items() if v is not None}

    def render(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        add_generation_prompt: bool = True,
        tools: Sequence[Mapping[str, Any]] | None = None,
        date: _dt.date | None = None,
        **variables: Any,
    ) -> str:
        """The prompt text for ``messages`` (dicts with ``role``, ``content`` and optionally ``tool_calls``)."""

        context = {
            **self._special_tokens,
            **variables,
            "messages": [dict(m) for m in messages],
            "add_generation_prompt": add_generation_prompt,
            "tools": [dict(t) for t in tools] if tools else None,
        }
        if date is not None:
            context["strftime_now"] = date.strftime
        try:
            return self._template.render(**context)
        except jinja2.TemplateError as error:
            raise ChatTemplateError(f"chat template failed: {error}") from error
