"""Schema algebra: ``allOf``, ``not`` and ``if``/``then``/``else`` rewritten into schemas the compiler enforces.

Constrained decoding (:mod:`etalii_dllm.grammar`) compiles a schema to a byte automaton. The combinators have no
automaton of their own; instead they are rewritten exactly, before compiling:

- ``allOf`` (and a schema's own keywords next to ``allOf``) is *merged* keyword by keyword into one schema: types
  intersect (``integer`` within ``number``), ``enum``/``const`` values intersect, bounds take the tighter value,
  ``multipleOf`` takes the exact least common multiple, every ``pattern`` applies, properties merge per name (a name
  one side does not declare gets that side's ``patternProperties`` or ``additionalProperties`` schema), tuples merge
  per position, and ``anyOf``/``oneOf`` branches distribute over the rest. Subschemas merge lazily (as a nested
  ``allOf``), so recursive schemas stay finite.
- ``not`` is *negated*: a schema is the conjunction of its keywords, so its negation is the ``anyOf`` of the negated
  keywords. Negatable are types (except ``integer`` alone), ``enum``/``const``, string lengths, patterns and formats,
  number bounds, ``multipleOf``, and object conditions made of ``properties`` and ``required``; ``anyOf``, ``allOf``
  and ``not`` negate by De Morgan's laws. Values, patterns, formats and divisors that a negation excludes are kept as
  internal exclusions (a ``not`` holding a tuple, which JSON cannot produce) that the compiler applies by automaton
  difference.
- ``if``/``then``/``else`` becomes ``anyOf`` of (``if`` and ``then``) and (``not if`` and ``else``).

A rewrite that cannot be exact is refused with a :class:`etalii_dllm.grammar.GrammarError`; an unsatisfiable merge
is ``False``. Every step is a pure function of the schemas, with keys and branches kept in the order the schema
gives them, so the rewritten schema (and the automaton) is the same on every machine.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from typing import Any

from etalii_dllm.grammar import _ANNOTATIONS, FORMATS, GrammarError, _identity, _searched

Schema = dict[str, Any] | bool
Resolve = Callable[[str], Any]

MAX_BRANCHES = 64
"""Most ``anyOf`` branches a merge may distribute into."""
_MAX_DEPTH = 32
_TYPES = ("null", "boolean", "object", "array", "string", "number")
_OBJECT_KEYS = ("properties", "required", "additionalProperties", "patternProperties", "propertyNames",
                "minProperties", "maxProperties")  # fmt: skip
_ARRAY_KEYS = ("items", "prefixItems", "additionalItems", "minItems", "maxItems", "uniqueItems", "contains",
               "minContains", "maxContains")  # fmt: skip
_LOWER = ("minLength", "minimum", "exclusiveMinimum")
_UPPER = ("maxLength", "maximum", "exclusiveMaximum")
_DROPPED = frozenset({"$defs", "definitions", "then", "else"})


def canonical(schema: Any) -> str:
    """A text equal for equal schemas (keys sorted), used to tell merges apart."""
    return json.dumps(schema, sort_keys=True, default=str, ensure_ascii=False)


def all_of(*schemas: Any) -> Any:
    """The lazy conjunction of ``schemas``: ``True`` and repeats dropped, ``False`` when one is ``False``."""
    kept: dict[str, Any] = {}
    for schema in schemas:
        if schema is False:
            return False
        if schema is True or schema == {}:
            continue
        kept.setdefault(canonical(schema), schema)
    if not kept:
        return True
    if len(kept) == 1:
        return next(iter(kept.values()))
    return {"allOf": list(kept.values())}


def searches(pattern: str, name: str) -> bool:
    from etalii_dllm.regexp import compile_regex

    return compile_regex(_searched(pattern)).matches(name.encode("utf-8"))


class Algebra:
    """Merges and negates the schemas of one root schema (``resolve`` looks up a local ``$ref``)."""

    def __init__(self, resolve: Resolve) -> None:
        self._resolve = resolve
        self._depth = 0

    # -- simplification -------------------------------------------------------------------------------------------

    def simplify(self, schema: Any) -> Schema:
        """``schema`` without ``$ref``, ``allOf``, ``not`` (a schema), ``if`` or ``nullable`` at the top: a plain
        schema, ``{"anyOf": [plain schemas]}`` or ``False``."""
        if schema is True:
            return {}
        if schema is False:
            return False
        if not isinstance(schema, Mapping):
            raise GrammarError(f"a schema must be an object, got {schema!r}")
        self._enter()
        try:
            return self._simplify(dict(schema))
        finally:
            self._depth -= 1

    def _enter(self) -> None:
        self._depth += 1
        if self._depth > _MAX_DEPTH:
            self._depth = 0
            raise GrammarError("the schema combinators nest too deeply (a recursive allOf, not or if?)")

    def _simplify(self, schema: dict[str, Any]) -> Schema:
        if schema.get("nullable") is True:  # OpenAPI: null as well
            rest = {k: v for k, v in schema.items() if k != "nullable"}
            return self._either([self.simplify(rest), {"type": "null"}])
        base = {k: v for k, v in schema.items() if k not in _DROPPED and k not in ("$ref", "allOf", "anyOf", "oneOf")}
        base.pop("nullable", None)
        parts: list[Schema] = []
        if "const" in base:
            values = [base.pop("const")]
            if "enum" in base:
                values = self._merge_keyword("enum", values, base["enum"])
                if values is False:
                    return False
            base["enum"] = values
        for bound, flag in (("minimum", "exclusiveMinimum"), ("maximum", "exclusiveMaximum")):
            if isinstance(base.get(flag), bool):  # draft 4: a boolean next to minimum/maximum
                exclusive = base.pop(flag)
                if exclusive and bound in base:
                    base[flag] = base.pop(bound)
        if isinstance(base.get("items"), list):  # draft 2019 and older: an items array is a tuple
            if "prefixItems" not in base:
                base["prefixItems"] = base["items"]
                base["items"] = base.get("additionalItems", True)
            base.pop("additionalItems", None)
        else:
            base.pop("additionalItems", None)
        negated = base.pop("not", None)
        if negated is not None and not isinstance(negated, tuple):
            parts.append(self.negate(negated))
        elif negated is not None:
            base["not"] = negated
        condition = base.pop("if", None)
        if condition is not None:
            parts.append(self.conditional(condition, schema.get("then", True), schema.get("else", True)))
        base = {k: v for k, v in base.items() if k not in _ANNOTATIONS or (k == "format" and v in FORMATS)}
        if "$ref" in schema:
            parts.append(self.simplify(self._resolve(str(schema["$ref"]))))
        for sub in schema.get("allOf") or ():
            parts.append(self.simplify(sub))
        for keyword in ("anyOf", "oneOf"):
            if keyword in schema:
                parts.append(self._either([self.simplify(sub) for sub in schema[keyword]]))
        result: Schema = base
        for part in parts:
            result = self.merge(result, part)
            if result is False:
                return False
        return result

    @staticmethod
    def _either(branches: Sequence[Schema]) -> Schema:
        kept: dict[str, Schema] = {}
        for branch in branches:
            if branch is False:
                continue
            if branch == {}:
                return {}
            for option in branch["anyOf"] if set(branch) == {"anyOf"} else [branch]:  # type: ignore[index]
                kept.setdefault(canonical(option), option)
        if not kept:
            return False
        if len(kept) == 1:
            return next(iter(kept.values()))
        if len(kept) > MAX_BRANCHES:
            raise GrammarError(f"the schema combinators expand into more than {MAX_BRANCHES} alternatives")
        return {"anyOf": list(kept.values())}

    def conditional(self, condition: Any, then: Any, otherwise: Any) -> Schema:
        """``if``/``then``/``else`` as ``anyOf`` of (``if`` and ``then``) and (``not if`` and ``else``)."""
        positive = self.merge(self.simplify(condition), self.simplify(then))
        negative = self.merge(self.negate(condition), self.simplify(otherwise))
        return self._either([positive, negative])

    # -- merging --------------------------------------------------------------------------------------------------

    def merge(self, first: Schema, second: Schema) -> Schema:
        """The conjunction of two simplified schemas, simplified."""
        if first is False or second is False:
            return False
        if first == {} or first == second:
            return second
        if second == {}:
            return first
        for one, other in ((first, second), (second, first)):
            if set(one) == {"anyOf"}:
                return self._either([self.merge(branch, other) for branch in one["anyOf"]])
        return self._merge_plain(first, second)

    def _merge_plain(self, first: dict[str, Any], second: dict[str, Any]) -> Schema:
        merged: dict[str, Any] = {}
        for key in dict.fromkeys([*first, *second]):
            if key in _OBJECT_KEYS or key in _ARRAY_KEYS:
                continue
            if key not in second or key not in first:
                merged[key] = first[key] if key in first else second[key]
                continue
            value = self._merge_keyword(key, first[key], second[key])
            if value is False and key in ("type", "enum"):
                return False
            merged[key] = value
        formats = [side["format"] for side in (first, second) if side.get("format") in FORMATS]
        if len(formats) == 2 and formats[0] != formats[1]:  # both apply: the second as an anchored pattern
            extra = f"^(?:{FORMATS[formats[1]]})$"
            merged["pattern"] = tuple(dict.fromkeys((*_tuple(merged.get("pattern", ())), extra)))
        if any(k in first or k in second for k in _OBJECT_KEYS):
            merged.update(self._merge_objects(first, second))
        if any(k in first or k in second for k in _ARRAY_KEYS):
            arrays = self._merge_arrays(first, second)
            merged.update(arrays)
        return merged

    def _merge_keyword(self, key: str, first: Any, second: Any) -> Any:
        if key == "type":
            return _intersect_types(first, second)
        if key == "enum":
            allowed = {_identity(value) for value in second}
            values = [value for value in first if _identity(value) in allowed]
            return values or False
        if key in _LOWER or key in _UPPER:
            pick = max if key in _LOWER else min
            return pick(first, second, key=lambda v: _exact(v, key))
        if key == "multipleOf":
            return _lcm(_exact(first, key), _exact(second, key))
        if key == "pattern":
            patterns = (*_tuple(first), *_tuple(second))
            return tuple(dict.fromkeys(patterns))
        if key == "format":
            return first if first in FORMATS or second not in FORMATS else second
        if key == "not":
            return (*first, *second)
        if key in _ANNOTATIONS:
            return first
        if first == second:
            return first
        raise GrammarError(f"'allOf' cannot merge two different values of '{key}'")

    def _merge_objects(self, first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        names = dict.fromkeys([*(first.get("properties") or {}), *(second.get("properties") or {})])
        if names:
            out["properties"] = {name: all_of(_member(first, name), _member(second, name)) for name in names}
        required = dict.fromkeys([*(first.get("required") or ()), *(second.get("required") or ())])
        if required:
            out["required"] = list(required)
        patterns = (first.get("patternProperties"), second.get("patternProperties"))
        additional = (first.get("additionalProperties", True), second.get("additionalProperties", True))
        if patterns[0] and patterns[1]:
            if canonical(patterns[0]) != canonical(patterns[1]) or canonical(additional[0]) != canonical(additional[1]):
                raise GrammarError("'allOf' cannot merge two schemas that both have 'patternProperties'")
            out["patternProperties"], merged_additional = patterns[0], additional[0]
        elif patterns[0] or patterns[1]:
            side = 0 if patterns[0] else 1
            out["patternProperties"] = {
                pattern: all_of(sub, additional[1 - side]) for pattern, sub in patterns[side].items()
            }
            merged_additional = all_of(*additional)
        else:
            merged_additional = all_of(*additional)
        if merged_additional is not True:
            out["additionalProperties"] = merged_additional
        if "propertyNames" in first or "propertyNames" in second:
            out["propertyNames"] = all_of(first.get("propertyNames", True), second.get("propertyNames", True))
        for key, pick in (("minProperties", max), ("maxProperties", min)):
            values = [side[key] for side in (first, second) if key in side]
            if values:
                out[key] = pick(values)
        return out

    def _merge_arrays(self, first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        prefixes = (first.get("prefixItems") or [], second.get("prefixItems") or [])
        rests = (first.get("items", True), second.get("items", True))
        maximum = [side["maxItems"] for side in (first, second) if "maxItems" in side]
        if prefixes[0] or prefixes[1]:
            merged_prefix = []
            for position in range(max(len(prefixes[0]), len(prefixes[1]))):
                elements = [p[position] if position < len(p) else r for p, r in zip(prefixes, rests, strict=True)]
                element = all_of(*elements)
                if element is False:
                    maximum.append(position)
                    break
                merged_prefix.append(element)
            out["prefixItems"] = merged_prefix
        rest = all_of(*rests)
        if rest is not True:
            out["items"] = rest
        minimum = [side["minItems"] for side in (first, second) if "minItems" in side]
        if minimum:
            out["minItems"] = max(minimum)
        if maximum:
            out["maxItems"] = min(maximum)
        if first.get("uniqueItems") is True or second.get("uniqueItems") is True:
            out["uniqueItems"] = True
        contains = [side for side in (first, second) if "contains" in side]
        if len(contains) == 2 and canonical(contains[0]["contains"]) != canonical(contains[1]["contains"]):
            raise GrammarError("'allOf' cannot merge two different 'contains' schemas")
        if contains:
            out["contains"] = contains[0]["contains"]
            least = [side["minContains"] for side in contains if "minContains" in side]
            most = [side["maxContains"] for side in contains if "maxContains" in side]
            if least:
                out["minContains"] = max(least)
            if most:
                out["maxContains"] = min(most)
        return out

    # -- negation -------------------------------------------------------------------------------------------------

    def negate(self, schema: Any) -> Schema:
        """The schema of the values ``schema`` rejects, simplified; refused when that is not exact."""
        simple = self.simplify(schema)
        if simple is False:
            return {}
        if simple == {}:
            return False
        self._enter()
        try:
            if set(simple) == {"anyOf"}:
                result: Schema = {}
                for branch in simple["anyOf"]:
                    result = self.merge(result, self._negate_plain(branch))
                return result
            return self._negate_plain(simple)
        finally:
            self._depth -= 1

    def _negate_plain(self, schema: Mapping[str, Any]) -> Schema:
        branches: list[Schema] = []
        for key, value in schema.items():
            if key in _ANNOTATIONS and not (key == "format" and value in FORMATS):
                continue
            if key in ("properties", "required"):
                continue
            branches += self._negate_keyword(key, value, schema)
        if "properties" in schema or "required" in schema:
            for name in schema.get("required") or ():
                branches.append({"type": "object", "properties": {name: False}})
            for name, sub in (schema.get("properties") or {}).items():
                negated = self.negate(sub)
                if negated is not False:
                    branches.append({"type": "object", "properties": {name: negated}, "required": [name]})
        return self._either(branches)

    def _negate_keyword(self, key: str, value: Any, schema: Mapping[str, Any]) -> list[Schema]:
        if key == "type":
            types = [value] if isinstance(value, str) else list(value)
            if "integer" in types and "number" not in types:
                raise GrammarError("'not' and 'if' cannot negate type 'integer' (the other numbers have no schema)")
            rest = [name for name in _TYPES if name not in types]
            return [{"type": rest}] if rest else []
        if key in ("enum", "const"):
            values = tuple(value) if key == "enum" else (value,)
            return [{"not": (("values", values),)}]
        if key == "minLength":
            return [{"type": "string", "maxLength": int(value) - 1}] if int(value) > 0 else []
        if key == "maxLength":
            return [{"type": "string", "minLength": int(value) + 1}]
        if key == "pattern":
            return [{"type": "string", "not": (("pattern", p),)} for p in _tuple(value)]
        if key == "format":
            return [{"type": "string", "not": (("format", value),)}]
        flipped = {"minimum": "exclusiveMaximum", "exclusiveMinimum": "maximum", "maximum": "exclusiveMinimum",
                   "exclusiveMaximum": "minimum"}  # fmt: skip
        if key in flipped:
            return [{"type": "number", flipped[key]: value}]
        if key == "multipleOf":
            return [{"type": "number", "not": (("multipleOf", value),)}]
        if key == "not":  # internal exclusions: excluding them again gives the values back
            branches: list[Schema] = []
            for kind, payload in value:
                if kind == "values":
                    branches.append({"enum": list(payload)})
                elif kind == "multipleOf":
                    branches.append({"type": "number", "multipleOf": payload})
                else:
                    branches.append({"type": "string", kind: payload})
            return branches
        raise GrammarError(f"'not' and 'if' cannot negate '{key}'")


def _member(schema: Mapping[str, Any], name: str) -> Any:
    """The schema ``schema`` gives a property called ``name``."""
    properties = schema.get("properties") or {}
    if name in properties:
        return properties[name]
    matched = [sub for pattern, sub in (schema.get("patternProperties") or {}).items() if searches(pattern, name)]
    if matched:
        return all_of(*matched)
    return schema.get("additionalProperties", True)


def _tuple(value: Any) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _exact(value: Any, name: str) -> Decimal:
    if isinstance(value, Decimal):
        return value
    from etalii_dllm.numeric_automata import exact

    return exact(value, name)


def _lcm(first: Decimal, second: Decimal) -> Decimal:
    """The least common multiple of two positive decimals, exactly."""
    if first <= 0 or second <= 0:
        raise GrammarError("'multipleOf' must be greater than 0")
    scale = max(-first.normalize().as_tuple().exponent, -second.normalize().as_tuple().exponent, 0)  # type: ignore[operator]
    a, b = int(first.scaleb(scale)), int(second.scaleb(scale))
    return Decimal(math.lcm(a, b)).scaleb(-scale)


def _intersect_types(first: Any, second: Any) -> Any:
    def names(value: Any) -> list[str]:
        return [value] if isinstance(value, str) else [str(v) for v in value]

    others = names(second)
    kept: list[str] = []
    for name in names(first):
        if name in others:
            kept.append(name)
        elif name in ("number", "integer") and ("integer" if name == "number" else "number") in others:
            kept.append("integer")  # integers are numbers
    kept = list(dict.fromkeys(kept))
    if not kept:
        return False
    return kept[0] if len(kept) == 1 else kept
