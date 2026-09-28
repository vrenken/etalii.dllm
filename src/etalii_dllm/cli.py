"""``dllm`` command line tool."""

from __future__ import annotations

import argparse
import sys

from etalii_dllm.engine import default_engine
from etalii_dllm.sampling import SamplingOptions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dllm", description="EtAlii deterministic LLM")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="show the model id and system fingerprint")

    generate = commands.add_parser("generate", help="continue a prompt")
    generate.add_argument("--prompt", default="")
    generate.add_argument("--max-tokens", type=int, default=64)
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-k", type=int, default=0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)
    engine = default_engine()

    if args.command == "info":
        print(f"model:              {engine.model.id}")
        print(f"system_fingerprint: {engine.system_fingerprint}")
        return 0

    options = SamplingOptions(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    result = engine.complete(args.prompt, args.max_tokens, options)
    print(result.text)
    print(
        f"fingerprint: {result.fingerprint}  tokens: {len(result.tokens)}  finish: {result.finish_reason}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
