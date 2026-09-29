"""``dllm`` command line tool."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from etalii_dllm import cuda
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import (
    ChatRequest,
    DllmEngine,
    Finished,
    ResponseFormat,
    TextDelta,
    add_runtime_arguments,
    default_engine,
    use_model_file,
)
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
    if model.fine_tuning:
        tuning = model.fine_tuning
        print(
            f"fine-tuned:         {tuning['steps_completed']} steps from {tuning['base_fingerprint'][:16]}"
            f" on data {tuning['data_fingerprint'][:16]}, final loss {tuning['final_loss']}"
        )
    print(f"system_fingerprint: {model.fingerprint}")
    return 0


def _finetune(args: argparse.Namespace) -> int:
    from etalii_dllm.engine import DllmEngine
    from etalii_dllm.modelfile import ModelFile, ModelFileError
    from etalii_dllm.training import (
        AdamWConfig,
        CheckpointError,
        FineTuner,
        RunConfig,
        StepResult,
        TrainingData,
        TrainingDataError,
        read_documents,
    )

    try:
        base = ModelFile(args.base)
        engine = DllmEngine.from_model_file(args.base, verify=False)
        render = None
        if engine.chat_template is not None:
            template = engine.chat_template
            render = lambda messages: template.render(messages, add_generation_prompt=False)  # noqa: E731
        separator = base.config.eos_token_ids[0] if base.config.eos_token_ids else None
        documents = read_documents(args.data, render)
        data = TrainingData.from_documents(documents, engine.tokenizer.encode, args.sequence_length, separator)
        if args.resume:
            tuner = FineTuner.load_checkpoint(args.resume, data)
        else:
            optimizer = AdamWConfig(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                warmup_steps=args.warmup_steps,
                schedule=args.schedule,
                min_learning_rate=args.min_learning_rate,
            )
            run = RunConfig(args.steps, args.batch_size, args.sequence_length, args.seed, optimizer)
            tuner = FineTuner.from_model_file(base, data, run)
    except (ModelFileError, TrainingDataError, CheckpointError, OSError, ValueError) as error:
        print(f"dllm finetune: {error}", file=sys.stderr)
        return 1

    print(f"data:               {len(data)} windows of up to {data.sequence_length} tokens, {data.fingerprint[:16]}")
    total = tuner.run.steps

    def report(result: StepResult) -> None:
        print(
            f"step {result.step:>5}/{total}  loss {result.loss:.6f}  lr {result.learning_rate:.3e}"
            f"  grad_norm {result.gradient_norm:.4f}"
        )
        if args.checkpoint and args.checkpoint_every and result.step % args.checkpoint_every == 0:
            tuner.save_checkpoint(args.checkpoint)

    tuner.train(on_step=report)
    if args.checkpoint:
        print(f"checkpoint:         {args.checkpoint} ({tuner.save_checkpoint(args.checkpoint)[:16]})")
    fingerprint = tuner.export(args.output)
    print(f"wrote:              {args.output}")
    print(f"system_fingerprint: {fingerprint}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dllm", description="EtAlii deterministic LLM")
    add_runtime_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="show the model id and system fingerprint")

    generate = commands.add_parser("generate", help="continue a prompt")
    generate.add_argument("--prompt", default="")
    chat = commands.add_parser("chat", help="answer a message using the model's chat template")
    chat.add_argument("message")
    chat.add_argument("--system", help="system message")
    chat.add_argument("--json", action="store_true", help="answer with a JSON object (constrained decoding)")
    chat.add_argument("--json-schema", help="answer with JSON valid under this schema (a file or inline JSON)")
    chat.add_argument("--mcp-config", help="let the model call the tools of these MCP servers ({'mcpServers': ...})")
    chat.add_argument(
        "--mcp-server",
        action="append",
        default=[],
        metavar="[NAME=]COMMAND|URL",
        help="an MCP server whose tools the model may call (repeatable), e.g. 'time=uvx mcp-server-time'",
    )
    chat.add_argument("--max-tool-rounds", type=int, default=8, help="MCP tool rounds before the answer is cut off")
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

    finetune = commands.add_parser("finetune", help="fine-tune a model.dllm reproducibly (AdamW, fixed data order)")
    finetune.add_argument("base", help="the model.dllm file to start from")
    finetune.add_argument("--data", required=True, help=".txt file, or .jsonl with {'text'} or {'messages'} lines")
    finetune.add_argument("-o", "--output", required=True, help="the fine-tuned model.dllm file to write")
    finetune.add_argument("--steps", type=int, default=100)
    finetune.add_argument("--batch-size", type=int, default=8)
    finetune.add_argument("--sequence-length", type=int, default=128)
    finetune.add_argument("--learning-rate", type=float, default=1e-4)
    finetune.add_argument("--min-learning-rate", type=float, default=0.0)
    finetune.add_argument("--warmup-steps", type=int, default=0)
    finetune.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    finetune.add_argument("--weight-decay", type=float, default=0.01)
    finetune.add_argument("--max-grad-norm", type=float, default=1.0, help="0 disables clipping")
    finetune.add_argument("--seed", type=int, default=0, help="seeds the data order")
    finetune.add_argument("--checkpoint", help="checkpoint file to write (at the end and every --checkpoint-every)")
    finetune.add_argument("--checkpoint-every", type=int, default=0)
    finetune.add_argument("--resume", help="continue from this checkpoint (run settings come from it)")

    args = parser.parse_args(argv)
    if args.command == "finetune":
        return _finetune(args)
    if args.command == "import":
        return _import(args)
    if args.command == "inspect":
        return _inspect(args)
    use_model_file(args.model, args.quantize, args.threads, args.device, args.prompt_cache)
    try:
        engine = default_engine()
    except cuda.CudaUnavailableError as error:
        print(f"dllm: --device cuda: {error}", file=sys.stderr)
        return 1

    if args.command == "info":
        print(f"model:              {engine.model.id}")
        print(f"system_fingerprint: {engine.system_fingerprint}")
        device = getattr(engine.model, "device", "cpu")
        gpu = cuda.info() if device == "cuda" else None
        print(f"device:             {device}" + (f" ({gpu.name}, {gpu.architecture}, {gpu.compiler})" if gpu else ""))
        return 0

    options = SamplingOptions(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, seed=args.seed)
    if args.command == "chat":
        return _chat(engine, args, options)
    generation = engine.complete_stream(args.prompt, args.max_tokens, options)
    for step in generation:
        _write(step.text)
    result = generation.result()
    _write("\n")
    print(
        f"fingerprint: {result.fingerprint}  tokens: {len(result.tokens)}  finish: {result.finish_reason}",
        file=sys.stderr,
    )
    return 0


def _write(text: str) -> None:
    """Prints generated text as it arrives."""
    sys.stdout.write(text)
    sys.stdout.flush()


def _chat(engine: DllmEngine, args: argparse.Namespace, options: SamplingOptions) -> int:
    response_format = ResponseFormat("json_object") if args.json else ResponseFormat()
    if args.json_schema:
        source = args.json_schema
        try:
            text = Path(source).read_text(encoding="utf-8") if not source.lstrip().startswith("{") else source
            response_format = ResponseFormat("json_schema", json.loads(text))
        except (OSError, json.JSONDecodeError) as error:
            print(f"dllm chat: --json-schema: {error}", file=sys.stderr)
            return 1
    messages = [ChatMessage("system", args.system)] if args.system else []
    request = ChatRequest(
        [*messages, ChatMessage("user", args.message)], args.max_tokens, options, response_format=response_format
    )
    if args.mcp_config or args.mcp_server:
        return _chat_with_mcp(engine, args, request)
    try:
        stream = engine.chat_stream(request)
    except ValueError as error:
        print(f"dllm chat: {error}", file=sys.stderr)
        return 1
    for event in stream:
        if isinstance(event, TextDelta):
            _write(event.text)
        elif isinstance(event, Finished):
            _write("\n")
            print(
                f"fingerprint: {event.fingerprint}  tokens: {event.completion_tokens}  finish: {event.finish_reason}",
                file=sys.stderr,
            )
    return 0


def _chat_with_mcp(engine: DllmEngine, args: argparse.Namespace, request: ChatRequest) -> int:
    import anyio

    from etalii_dllm import mcp_host
    from etalii_dllm.engine import ToolCallEvent

    async def run() -> int:
        finished: Finished | None = None
        servers = mcp_host.load_config(args.mcp_config) if args.mcp_config else []
        servers += [mcp_host.parse_server(spec) for spec in args.mcp_server]
        async with mcp_host.McpHost(servers) as host:
            print(f"tools: {', '.join(t.name for t in host.tools) or '(none)'}", file=sys.stderr)
            async for event in mcp_host.chat(engine, request, host, args.max_tool_rounds):
                if isinstance(event, TextDelta):
                    _write(event.text)
                elif isinstance(event, ToolCallEvent):
                    print(f"\n-> {event.call.name}({event.call.arguments})", file=sys.stderr)
                elif isinstance(event, mcp_host.ToolResult):
                    marker = "error" if event.is_error else "result"
                    print(f"<- {marker}: {event.content}", file=sys.stderr)
                else:
                    finished = event
        assert finished is not None
        _write("\n")
        print(
            f"fingerprint: {finished.fingerprint}  tokens: {finished.completion_tokens}"
            f"  finish: {finished.finish_reason}",
            file=sys.stderr,
        )
        return 0

    try:
        return anyio.run(run)
    except (mcp_host.McpHostError, ValueError) as error:
        print(f"dllm chat: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
