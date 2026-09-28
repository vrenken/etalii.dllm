"""``dllm`` command line tool."""

from __future__ import annotations

import argparse
import json
import sys

from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import default_engine, use_model_file
from etalii_dllm.sampling import SamplingOptions


def _import(args: argparse.Namespace) -> int:
    from etalii_dllm.importing import ModelImportError, import_model

    try:
        result = import_model(
            args.source,
            args.output,
            repository=args.repo,
            revision=args.revision,
            licence=args.licence,
            licence_file=args.licence_file,
            accept_licence=args.accept_licence,
            cache=args.cache,
        )
    except (ModelImportError, OSError) as error:
        print(f"dllm import: {error}", file=sys.stderr)
        return 1
    config = result.config
    print(f"wrote:              {result.path}")
    print(f"architecture:       {config.family}, {config.layers} layers, hidden {config.hidden_size}")
    print(f"licence:            {result.licence['spdx']}")
    print(f"system_fingerprint: {result.fingerprint}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    from etalii_dllm.modelfile import ModelFile, ModelFileError

    try:
        model = ModelFile(args.path, verify=not args.no_verify)
    except (ModelFileError, OSError) as error:
        print(f"dllm inspect: {error}", file=sys.stderr)
        return 1
    parameters = sum(int(t.size) for t in model.tensors.values())
    print(f"architecture:       {json.dumps(model.config.to_dict(), sort_keys=True)}")
    print(f"parameters:         {parameters}")
    print(f"source:             {json.dumps({k: v for k, v in model.source.items() if k != 'files'}, sort_keys=True)}")
    print(f"licence:            {model.licence.get('spdx')}")
    print(f"attribution:        {model.licence.get('attribution')}")
    print(f"chat template:      {'yes' if model.chat_template else 'no'}")
    print(f"system_fingerprint: {model.fingerprint}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dllm", description="EtAlii deterministic LLM")
    parser.add_argument("--model", help="model.dllm file to use (default: $DLLM_MODEL, else the placeholder model)")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="show the model id and system fingerprint")

    generate = commands.add_parser("generate", help="continue a prompt")
    generate.add_argument("--prompt", default="")
    chat = commands.add_parser("chat", help="answer a message using the model's chat template")
    chat.add_argument("message")
    chat.add_argument("--system", help="system message")
    for command in (generate, chat):
        command.add_argument("--max-tokens", type=int, default=64 if command is generate else 256)
        command.add_argument("--temperature", type=float, default=0.0)
        command.add_argument("--top-k", type=int, default=0)
        command.add_argument("--top-p", type=float, default=1.0)
        command.add_argument("--seed", type=int, default=0)

    importer = commands.add_parser("import", help="convert an open-weight model to model.dllm")
    importer.add_argument("source", help="checkpoint directory, .gguf file, or hf:org/name[@revision]")
    importer.add_argument("-o", "--output", required=True, help="the model.dllm file to write")
    importer.add_argument("--repo", help="source repository to record (local sources)")
    importer.add_argument("--revision", help="source revision to record (local sources)")
    importer.add_argument("--licence", help="SPDX id, when the source does not state its licence")
    importer.add_argument("--licence-file", help="licence text, when the source does not include it")
    importer.add_argument("--accept-licence", action="store_true", help="import a model that is not Apache/MIT")
    importer.add_argument("--cache", help="download cache for hf: sources")

    inspect = commands.add_parser("inspect", help="show a model.dllm file's architecture, source and licence")
    inspect.add_argument("path")
    inspect.add_argument("--no-verify", action="store_true", help="skip re-hashing the tensor data")

    args = parser.parse_args(argv)
    if args.command == "import":
        return _import(args)
    if args.command == "inspect":
        return _inspect(args)
    use_model_file(args.model)
    engine = default_engine()

    if args.command == "info":
        print(f"model:              {engine.model.id}")
        print(f"system_fingerprint: {engine.system_fingerprint}")
        return 0

    options = SamplingOptions(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    if args.command == "chat":
        messages = [ChatMessage("system", args.system)] if args.system else []
        result = engine.chat([*messages, ChatMessage("user", args.message)], args.max_tokens, options)
    else:
        result = engine.complete(args.prompt, args.max_tokens, options)
    print(result.text)
    print(
        f"fingerprint: {result.fingerprint}  tokens: {len(result.tokens)}  finish: {result.finish_reason}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
