"""``dllm`` command line tool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from etalii_dllm import cuda
from etalii_dllm.chat import ChatMessage
from etalii_dllm.engine import (
    MAX_CHOICES,
    MODEL_ENVIRONMENT_VARIABLE,
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
    from etalii_dllm.importing.gguf import GgufError
    from etalii_dllm.importing.safetensors import SafetensorsError
    from etalii_dllm.modelfile import file_sha256

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
            base=args.base,
        )
    except (ModelImportError, GgufError, SafetensorsError, OSError) as error:
        print(f"dllm import: {error}", file=sys.stderr)
        return 1
    config = result.config
    print(f"wrote:              {result.path}")
    print(f"architecture:       {config.family}, {config.layers} layers, hidden {config.hidden_size}")
    print(f"licence:            {result.licence['spdx']}")
    print(f"system_fingerprint: {result.fingerprint}")
    print(f"file_sha256:        {file_sha256(result.path)}")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    from etalii_dllm.modelfile import ModelFile, ModelFileError, lineage_problems

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
    if model.adapter:
        lora = model.adapter["lora"]
        print(
            f"adapter:            LoRA rank {lora['rank']} alpha {lora['alpha']} on {','.join(lora['targets'])},"
            f" merged into {model.adapter['base_fingerprint'][:16]}"
        )
    for edit in model.edits:
        print(
            f"edited:             {edit['method']} at layer {edit['layer']}: {edit['prompt']!r} -> {edit['target']!r}"
            f" (from {edit['base_fingerprint'][:16]})"
        )
    merge = model.header.get("merge")
    if merge:
        inputs = ", ".join(str(item["fingerprint"])[:16] for item in merge["inputs"])
        print(f"merged:             {merge['method']} of {inputs}")
    for index, step in enumerate(model.lineage):
        details = ", ".join(f"{k} {str(v)[:16]}" for k, v in step.items() if k not in ("step", "input", "output"))
        print(f"lineage {index}:          {step['step']} ({details}) -> {str(step.get('output'))[:16]}")
    for problem in lineage_problems(model.lineage):
        print(f"lineage problem:    {problem}")
    print(f"system_fingerprint: {model.fingerprint}")
    if args.trust:
        from etalii_dllm import signing

        try:
            problem = signing.model_signature_problem(args.path, [signing.read_public_key(k) for k in args.trust])
        except signing.SigningError as error:
            print(f"dllm inspect: {error}", file=sys.stderr)
            return 1
        print(f"signature:          {problem or 'valid, by a trusted key'}")
        return 1 if problem else 0
    return 0


def _finetune(args: argparse.Namespace) -> int:
    from etalii_dllm.engine import DllmEngine
    from etalii_dllm.modelfile import ModelFile, ModelFileError
    from etalii_dllm.training import (
        AdamWConfig,
        CheckpointError,
        FineTuner,
        LoraConfig,
        RunConfig,
        StepResult,
        TrainingDataError,
    )
    from etalii_dllm.training.receipt import load_data, make_receipt

    if not args.output and not args.adapter_output:
        print("dllm finetune: pass -o/--output, --adapter-output, or both", file=sys.stderr)
        return 1
    distillation = None
    if args.command == "distill" or args.teacher:
        if not args.teacher or not args.prompts:
            print("dllm distill: pass --teacher MODEL and --prompts FILE", file=sys.stderr)
            return 1
        from etalii_dllm.training.distill import write_teacher_data

        args.data = args.data or f"{args.output or args.adapter_output}.distill.jsonl"
        try:
            distillation = write_teacher_data(args.teacher, args.prompts, args.data, args.teacher_max_tokens)
        except (ModelFileError, OSError, ValueError) as error:
            print(f"dllm distill: {error}", file=sys.stderr)
            return 1
        examples, teacher = distillation["examples"], distillation["teacher"][:16]
        print(f"teacher:            {examples} answers from {teacher} in {args.data}")
    elif not args.data:
        print("dllm finetune: pass --data FILE (or --teacher and --prompts to distill)", file=sys.stderr)
        return 1
    try:
        base = ModelFile(args.base)
        engine = DllmEngine.from_model_file(args.base, verify=False)
        data = load_data(engine, base, args.data, args.sequence_length)
        if args.resume:
            tuner = FineTuner.load_checkpoint(args.resume, data, base)
        else:
            optimizer = AdamWConfig(
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_grad_norm=args.max_grad_norm,
                warmup_steps=args.warmup_steps,
                schedule=args.schedule,
                min_learning_rate=args.min_learning_rate,
            )
            lora = None
            if args.lora_rank:
                targets = tuple(t.strip() for t in args.lora_targets.split(",") if t.strip())
                alpha = args.lora_alpha if args.lora_alpha is not None else float(args.lora_rank)
                lora = LoraConfig(args.lora_rank, alpha, targets)
            run = RunConfig(args.steps, args.batch_size, args.sequence_length, args.seed, optimizer, lora)
            tuner = FineTuner.from_model_file(base, data, run)
        tuner.distillation = distillation
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
    if args.adapter_output:
        if tuner.lora is None:
            print("dllm finetune: --adapter-output needs a LoRA run (--lora-rank)", file=sys.stderr)
            return 1
        tuner.export_adapter(args.adapter_output)
        print(f"adapter:            {args.adapter_output}")
    if args.output:
        fingerprint = tuner.export(args.output)
        print(f"wrote:              {args.output}")
        print(f"system_fingerprint: {fingerprint}")
    if args.receipt:
        receipt = make_receipt(tuner, args.data)
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        print(f"receipt:            {args.receipt} ({receipt['id']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dllm", description="EtAlii deterministic LLM")
    add_runtime_arguments(parser)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="show the model id and system fingerprint")
    verify = commands.add_parser("verify", help="one fingerprint to compare with another machine (same bits?)")
    verify.add_argument("--json", action="store_true", help="print the report as JSON")
    verify.add_argument(
        "--reference",
        action="store_true",
        help="also check the model's answers against the independent reference implementation (slow)",
    )

    generate = commands.add_parser("generate", help="continue a prompt")
    generate.add_argument("--prompt", default="")
    generate.add_argument(
        "--n", type=int, default=1, help="generate N choices; choice i samples with seed SEED+i (default 1)"
    )
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
    chat.add_argument(
        "--tool",
        action="append",
        default=[],
        metavar="NAME[=ARG]",
        help="a built-in deterministic tool the model may call (repeatable): calculator, files=DIR, documents",
    )
    chat.add_argument("--max-tool-rounds", type=int, default=8, help="MCP tool rounds before the answer is cut off")
    chat.add_argument("--receipt", metavar="FILE", help="write the answer's generation receipt to FILE (JSON)")
    chat.add_argument(
        "--transcript", metavar="FILE", help="with MCP tools: write the whole agent run to FILE (dllm replay checks it)"
    )
    for command in (generate, chat):
        command.add_argument("--max-tokens", type=int, default=64 if command is generate else 256)
        command.add_argument("--temperature", type=float, default=0.0)
        command.add_argument("--top-k", type=int, default=0)
        command.add_argument("--top-p", type=float, default=1.0)
        command.add_argument("--seed", type=int, default=0)
        command.add_argument("--min-p", type=float, default=0.0, help="drop tokens below MIN_P times the top one")
        command.add_argument("--repetition-penalty", type=float, default=1.0, help="penalise recent tokens (1: off)")
        command.add_argument(
            "--repeat-last-n", type=int, default=64, help="tokens the repetition penalty looks at (-1: all)"
        )
        command.add_argument("--frequency-penalty", type=float, default=0.0)
        command.add_argument("--presence-penalty", type=float, default=0.0)
        command.add_argument(
            "--logit-bias", action="append", default=[], metavar="TOKEN=BIAS", help="add BIAS to a token (repeatable)"
        )
        command.add_argument("--regex", help="only produce text matching this regular expression in full")

    evaluate = commands.add_parser("eval", help="score the model on a task file: perplexity or multiple choice")
    evaluate.add_argument("task", metavar="TASK", help=".jsonl ({'context','choices','answer'} or {'text'}) or .txt")
    evaluate.add_argument("--max-length", type=int, help="longest window for perplexity texts (default: 1024)")
    evaluate.add_argument("--json", action="store_true", help="print the full report, per-item results included")
    evaluate.add_argument("-o", "--output", help="also write the full report (JSON) to this file")

    replay = commands.add_parser("replay", help="re-run a generation receipt and check the output is the same")
    replay.add_argument(
        "receipt", metavar="FILE", help="a receipt, a receipt chain (JSON list) or agent transcript; - reads stdin"
    )
    replay.add_argument("--json", action="store_true", help="print the verification as JSON")
    replay.add_argument(
        "--trust",
        action="append",
        default=[],
        metavar="KEY",
        help="also require a valid signature by this Ed25519 public key (hex or .pub file; repeatable)",
    )
    replay.add_argument("--base", help="training receipts: the model.dllm the run started from (default: --model)")
    replay.add_argument("--data", help="training receipts: the training data (default: the file the receipt names)")
    replay.add_argument("--teacher", help="distillation receipts: regenerate the data with this teacher model.dllm")
    replay.add_argument("--prompts", help="distillation receipts: the prompts (default: the file the receipt names)")

    from etalii_dllm.interpret import commands as interpret_commands

    interpret_commands.add_commands(commands)

    index = commands.add_parser("index", help="build or search a document index (retrieval, embedding models)")
    index_commands = index.add_subparsers(dest="index_command", required=True)
    build = index_commands.add_parser("build", help="chunk and embed documents with --model into an index file")
    build.add_argument("paths", nargs="+", help="text files, or directories (their .md/.txt/... files)")
    build.add_argument("-o", "--output", required=True, help="the index file to write")
    build.add_argument("--chunk-tokens", type=int, default=256, help="largest chunk, in tokens")
    search = index_commands.add_parser("search", help="the passages closest to a query (exact, reproducible)")
    search.add_argument("index_file", metavar="INDEX", help="the index file")
    search.add_argument("query")
    search.add_argument("--top", type=int, default=5)
    search.add_argument("--json", action="store_true", help="print the hits as JSON")

    importer = commands.add_parser("import", help="convert an open-weight model to model.dllm")
    importer.add_argument("source", help="checkpoint directory, .gguf file, or hf:org/name[@revision]")
    importer.add_argument("-o", "--output", required=True, help="the model.dllm file to write")
    importer.add_argument("--repo", help="source repository to record (local sources)")
    importer.add_argument("--revision", help="source revision to record (local sources)")
    importer.add_argument("--licence", help="SPDX id, when the source does not state its licence")
    importer.add_argument("--licence-file", help="licence text, when the source does not include it")
    importer.add_argument("--accept-licence", action="store_true", help="import a model that is not Apache/MIT")
    importer.add_argument("--cache", help="download cache for hf: sources")
    importer.add_argument("--base", help="the model.dllm to merge a PEFT LoRA adapter source into")

    inspect = commands.add_parser("inspect", help="show a model.dllm file's architecture, source and licence")
    inspect.add_argument("path")
    inspect.add_argument("--no-verify", action="store_true", help="skip re-hashing the tensor data")
    inspect.add_argument(
        "--trust", action="append", default=[], metavar="KEY", help="check FILE.sig against this public key"
    )

    finetune = commands.add_parser(
        "finetune",
        aliases=["distill"],
        help="fine-tune a model.dllm reproducibly (AdamW, fixed data order); distill: on a --teacher's answers",
    )
    finetune.add_argument("base", help="the model.dllm file to start from")
    finetune.add_argument(
        "--data",
        help=".txt file, or .jsonl with {'text'} or {'messages'} lines (with --teacher: where to write its answers, "
        "default OUTPUT.distill.jsonl)",
    )
    finetune.add_argument("--teacher", help="distill: train on this model.dllm's greedy answers to --prompts")
    finetune.add_argument("--prompts", help="distill: one prompt per line (or a JSON object with 'messages')")
    finetune.add_argument("--teacher-max-tokens", type=int, default=256, help="distill: tokens per teacher answer")
    finetune.add_argument("-o", "--output", help="the fine-tuned model.dllm file to write (LoRA: adapters merged)")
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
    finetune.add_argument("--lora-rank", type=int, default=0, help="train LoRA adapters of this rank, not all weights")
    finetune.add_argument("--lora-alpha", type=float, help="LoRA scale numerator (default: the rank, so scale 1)")
    finetune.add_argument(
        "--lora-targets", default="q,k,v,o,gate,up,down", help="linear layers to adapt (comma separated)"
    )
    finetune.add_argument("--adapter-output", help="LoRA runs: write the adapters as a PEFT directory here")
    finetune.add_argument("--receipt", metavar="FILE", help="write a training receipt (dllm replay trains again)")

    sign = commands.add_parser("sign", help="sign a receipt, chain, transcript or model file with an Ed25519 key")
    sign.add_argument("file", nargs="?", help="the JSON document or model.dllm file to sign")
    sign.add_argument("--key", help="the private key file")
    sign.add_argument("-o", "--output", help="JSON documents: write the signed document here (default: in place)")
    sign.add_argument("--keygen", metavar="FILE", help="create a private key FILE and its public key FILE.pub")

    merge = commands.add_parser("merge", help="merge models of one architecture into a new model.dllm (exact)")
    merge.add_argument("models", nargs="+", help="the model.dllm files to merge (the first one's lineage continues)")
    merge.add_argument("-o", "--output", required=True, help="the merged model.dllm")
    merge.add_argument("--method", default="linear", choices=("linear", "slerp", "ties"))
    merge.add_argument("--weights", help="comma-separated weight per model (default: equal)")
    merge.add_argument("--t", type=float, default=0.5, help="slerp: 0 gives the first model, 1 the second")
    merge.add_argument("--base", help="ties: the model the others were fine-tuned from")
    merge.add_argument("--density", type=float, default=0.2, help="ties: share of each task vector to keep")

    conformance = commands.add_parser(
        "conformance", help="write or check the conformance vectors that prove an implementation's bits"
    )
    conformance_commands = conformance.add_subparsers(dest="conformance_command", required=True)
    conformance_write = conformance_commands.add_parser("write", help="write the vectors of this build")
    conformance_write.add_argument("directory")
    conformance_check = conformance_commands.add_parser("check", help="check an implementation against vectors")
    conformance_check.add_argument("directory")
    conformance_check.add_argument(
        "--implementation",
        default="kernels",
        choices=("kernels", "reference"),
        help="the compiled kernels (default) or the independent reference implementation",
    )

    export = commands.add_parser("export", help="write a model.dllm as Hugging Face safetensors or GGUF (exact)")
    export.add_argument("path", help="the model.dllm file")
    export.add_argument("--format", required=True, choices=("safetensors", "gguf"))
    export.add_argument("-o", "--output", required=True, help="a directory (safetensors) or a .gguf file")

    cache_command = commands.add_parser("cache", help="show or clear a response cache directory (--response-cache)")
    cache_commands = cache_command.add_subparsers(dest="cache_command", required=True)
    for name, text in (("stats", "how many responses DIR holds"), ("clear", "remove every response in DIR")):
        cache_commands.add_parser(name, help=text).add_argument("directory", metavar="DIR")

    audit = commands.add_parser("audit", help="check that servers (and this engine) give the same bits")
    audit.add_argument("--url", action="append", default=[], help="a server's base URL (repeatable)")
    audit.add_argument("--local", action="store_true", help="also run every request on this engine (--model ...)")
    audit.add_argument(
        "--prompts", required=True, help="one request per line: a user message, or a chat completions JSON object"
    )
    audit.add_argument("--max-tokens", type=int, default=32, help="for lines that do not set max_tokens")
    audit.add_argument("--json", action="store_true", help="print the results as JSON")

    args = parser.parse_args(argv)
    if args.command == "sign":
        return _sign(args)
    if args.command == "cache":
        return _cache(args)
    if args.command == "merge":
        return _merge(args)
    if args.command == "export":
        return _export(args)
    if args.command == "conformance":
        return _conformance(args)
    if args.command == "audit" and not args.local:
        return _audit(None, args)
    if args.command in ("finetune", "distill"):
        return _finetune(args)
    if args.command == "edit":
        return interpret_commands.run_edit(args)
    if args.command == "import":
        return _import(args)
    if args.command == "inspect":
        return _inspect(args)
    if args.command == "replay":
        try:
            args.loaded = _read_json(args.receipt)
        except (OSError, ValueError) as error:
            print(f"dllm replay: {error}", file=sys.stderr)
            return 2
        problems = _signature_problems(args.loaded, args.trust)
        if problems is None:
            return 2
        if isinstance(args.loaded, dict) and "training_receipt" in args.loaded:
            return _signed(_replay_training(args), problems)
    configured = args.model or os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if args.command == "index" and args.index_command == "search" and not configured:
        args.model = _index_model(args.index_file)  # search with the model the index was built with
    use_model_file(
        args.model,
        args.quantize,
        args.threads,
        args.device,
        args.prompt_cache,
        args.adapter,
        steer=args.steer,
        steer_strength=args.steer_strength,
        index=args.index,
        index_top=args.index_top,
        embedding_model=args.embedding_model,
        speculate=args.speculate,
        draft_model=args.draft_model,
        prompt_cache_dir=args.prompt_cache_dir,
        sign_key=args.sign_key,
        response_cache=args.response_cache,
        audit_every=args.audit_every,
    )
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

    if args.command == "verify":
        return _verify(engine, args.json, args.reference)

    if args.command == "audit":
        return _audit(engine, args)

    if args.command == "replay":
        return _signed(_replay(engine, args), problems)

    if args.command == "eval":
        return _evaluate(engine, args)

    if args.command in interpret_commands.COMMANDS:
        return interpret_commands.run(args, engine)
    if args.command == "index":
        return _index(args, engine)

    try:
        options = _sampling_options(args)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.command == "chat":
        return _chat(engine, args, options)
    if not 1 <= args.n <= MAX_CHOICES:
        print(f"error: --n must be between 1 and {MAX_CHOICES}", file=sys.stderr)
        return 2
    for index in range(args.n):
        if args.n > 1:
            print(f"--- choice {index} (seed {options.for_choice(index).seed})", file=sys.stderr)
        try:
            generation = engine.complete_stream(
                args.prompt, args.max_tokens, options.for_choice(index), regex=args.regex
            )
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        for step in generation:
            _write(step.text)
        result = generation.result()
        _write("\n")
        print(
            f"fingerprint: {result.fingerprint}  tokens: {len(result.tokens)}  finish: {result.finish_reason}",
            file=sys.stderr,
        )
    return 0


def _sampling_options(args: argparse.Namespace) -> SamplingOptions:
    bias: dict[int, float] = {}
    for item in args.logit_bias:
        token, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"--logit-bias expects TOKEN=BIAS, not {item!r}")
        bias[int(token)] = float(value)
    return SamplingOptions(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        min_p=args.min_p,
        repetition_penalty=args.repetition_penalty,
        repeat_last_n=args.repeat_last_n,
        frequency_penalty=args.frequency_penalty,
        presence_penalty=args.presence_penalty,
        logit_bias=SamplingOptions.bias(bias),
    )


def _index_model(path: str) -> str | None:
    from etalii_dllm.retrieval import Index, RetrievalError

    try:
        return Index.load(path).model.get("path")
    except RetrievalError:
        return None  # reported by the search itself


def _index(args: argparse.Namespace, engine: DllmEngine) -> int:
    from etalii_dllm import retrieval

    try:
        if args.index_command == "build":
            documents = retrieval.read_documents(args.paths)
            model_path = args.model or os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
            model_path = str(Path(model_path).resolve()) if model_path else None

            def progress(count: int, chunk: retrieval.Chunk) -> None:
                print(f"chunk {count:6d}  {chunk.source}:{chunk.start}", file=sys.stderr)

            index = retrieval.build_index(
                engine, documents, chunk_tokens=args.chunk_tokens, model_path=model_path, progress=progress
            )
            index.save(args.output)
            print(f"wrote: {args.output}  ({len(documents)} documents, {len(index.chunks)} chunks)")
            print(f"index fingerprint: {index.fingerprint}")
            return 0
        index = retrieval.Index.load(args.index_file)
        hits = index.search(engine, args.query, args.top)
    except (OSError, ValueError) as error:
        print(f"dllm index: {error}", file=sys.stderr)
        return 1
    if args.json:
        rows = [{"rank": h.rank, "score": h.score, "chunk": h.index, **h.chunk.to_json()} for h in hits]
        print(json.dumps(rows, indent=2, ensure_ascii=False))
        return 0
    for hit in hits:
        print(f"{hit.rank}. {hit.score:.4f}  {hit.chunk.source}:{hit.chunk.start}-{hit.chunk.end}")
        print("   " + hit.chunk.text.replace("\n", "\n   "))
    return 0


def _verify(engine: DllmEngine, as_json: bool, against_reference: bool = False) -> int:
    from etalii_dllm import verify

    report = verify.run(engine)
    check = None
    if against_reference:
        try:
            check = verify.check_reference(engine)
        except ValueError as error:
            print(f"dllm verify: {error}", file=sys.stderr)
            return 2
    failed = bool(report.mismatches) or (check is not None and not check.equal)
    if as_json:
        result = report.as_dict()
        if check is not None:
            result["reference"] = check.as_dict()
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for name, value in report.environment.items():
            print(f"{name + ':':<20}{value}")
        print()
        for name, value in report.parts.items():
            reference = verify.REFERENCE.get(name)
            note = "" if not reference else ("  (as released)" if value == reference else "  (DIFFERS from release)")
            print(f"{name + ':':<20}{value[:32]}{note}")
        print(f"\nverify:             {report.fingerprint}")
        print("Equal verify fingerprints (same model and options) mean the two machines give the same bits.")
        if check is not None:
            print("\nagainst the reference implementation:")
            for name, value in check.results.items():
                print(f"{name + ':':<20}{value}")
            if check.equal:
                print("This machine computes exactly what the specification says.")
    return 1 if failed else 0


def _evaluate(engine: DllmEngine, args: argparse.Namespace) -> int:
    from etalii_dllm import evaluation

    def progress(done: int, total: int) -> None:
        print(f"item {done}/{total}", file=sys.stderr)

    try:
        items = evaluation.read_task(args.task)
        report = evaluation.evaluate(
            engine, items, task=Path(args.task).name, max_length=args.max_length, progress=progress
        )
    except evaluation.EvaluationError as error:
        print(f"dllm eval: {error}", file=sys.stderr)
        return 1
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.json:
        print(text)
        return 0
    for key, value in report.items():
        if key != "results":
            print(f"{key + ':':<20}{value}")
    return 0


def _merge(args: argparse.Namespace) -> int:
    from etalii_dllm.merging import MergeError, merge_models
    from etalii_dllm.modelfile import ModelFileError, file_sha256

    try:
        weights = [float(w) for w in args.weights.split(",")] if args.weights else None
        fingerprint = merge_models(
            args.models, args.output, args.method, weights, base=args.base, t=args.t, density=args.density
        )
    except (MergeError, ModelFileError, OSError, ValueError) as error:
        print(f"dllm merge: {error}", file=sys.stderr)
        return 2
    print(f"wrote {args.output}")
    print(f"system_fingerprint: {fingerprint}")
    print(f"file_sha256:        {file_sha256(args.output)}")
    return 0


def _export(args: argparse.Namespace) -> int:
    from etalii_dllm.exporting import ExportError, export_model
    from etalii_dllm.modelfile import ModelFileError, file_sha256

    try:
        files = export_model(args.path, args.output, args.format)
    except (ExportError, ModelFileError, OSError) as error:
        print(f"dllm export: {error}", file=sys.stderr)
        return 2
    for path in files:
        print(f"{file_sha256(path)}  {path}")
    return 0


def _conformance(args: argparse.Namespace) -> int:
    from etalii_dllm import conformance

    if args.conformance_command == "write":
        count, digest = conformance.write(args.directory)
        print(f"wrote {count} cases to {args.directory}")
        print(f"manifest_sha256: {digest}")
        return 0
    try:
        result = conformance.check(args.directory, args.implementation)
    except (ValueError, OSError) as error:
        print(f"dllm conformance: {error}", file=sys.stderr)
        return 2
    for name, reason in result.failed.items():
        print(f"FAIL {name}: {reason}")
    print(f"manifest_sha256: {result.manifest_sha256}")
    print(f"{len(result.passed)} passed, {len(result.failed)} failed ({args.implementation})")
    return 0 if result.ok else 1


def _cache(args: argparse.Namespace) -> int:
    from etalii_dllm.serving import ResponseCache

    if not Path(args.directory).is_dir():
        print(f"dllm cache: {args.directory} is not a directory", file=sys.stderr)
        return 2
    cache = ResponseCache(args.directory)
    if args.cache_command == "clear":
        print(f"removed {cache.clear()} responses")
        return 0
    stats = cache.stats()
    print(f"responses: {stats['responses']}")
    print(f"bytes:     {stats['bytes']}")
    return 0


def _audit(engine: DllmEngine | None, args: argparse.Namespace) -> int:
    """``dllm audit``: the same requests to every server (and this engine), compared receipt by receipt."""
    from functools import partial

    from etalii_dllm import serving

    targets: dict[str, Any] = {url: partial(serving.post_json, url) for url in args.url}
    if engine is not None:
        from etalii_dllm.server.app import _chat_request
        from etalii_dllm.server.contracts import ChatCompletionRequest

        def local(body: Any) -> dict[str, Any]:
            request = _chat_request(ChatCompletionRequest.model_validate(body), engine)
            return {"receipt": engine.chat_completion(request, fresh=True).receipt}

        targets = {"local": local, **targets}
    if len(targets) < 2:
        print("dllm audit: give at least two targets (--url, --local)", file=sys.stderr)
        return 2
    try:
        lines = [line for line in Path(args.prompts).read_text(encoding="utf-8").splitlines() if line.strip()]
        bodies = [serving.request_body(line, args.max_tokens) for line in lines]
        results = serving.audit_servers(targets, bodies)
    except (OSError, ValueError, KeyError) as error:
        print(f"dllm audit: {error}", file=sys.stderr)
        return 2
    agree = all(result.ok for result in results)
    if args.json:
        rows = [{"ok": r.ok, "differences": list(r.differences), "outputs": r.outputs} for r in results]
        print(json.dumps({"ok": agree, "targets": list(targets), "requests": rows}, indent=2))
    else:
        for number, result in enumerate(results, 1):
            first = next(iter(result.outputs.values()))
            print(f"request {number}: {'same' if result.ok else 'DIFFERENT'} ({first['tokens']})")
            for difference in result.differences:
                print(f"  {difference}")
        print(f"{len(targets)} targets, {len(results)} requests: {'the same bits' if agree else 'they differ'}")
    return 0 if agree else 1


def _sign(args: argparse.Namespace) -> int:
    from etalii_dllm import signing
    from etalii_dllm.modelfile import MAGIC

    try:
        if args.keygen:
            print(f"public key:         {signing.generate_key(args.keygen)}  ({args.keygen}.pub)")
            return 0
        if not args.file or not args.key:
            print("dllm sign: pass FILE and --key KEY (or --keygen FILE)", file=sys.stderr)
            return 2
        signer = signing.Signer.load(args.key)
        with Path(args.file).open("rb") as stream:
            is_model = stream.read(len(MAGIC)) == MAGIC
        if is_model:
            target = Path(f"{args.file}.sig")
            target.write_text(json.dumps(signer.sign_model(args.file), indent=2) + "\n", encoding="utf-8")
        else:
            document = _read_json(args.file)
            if isinstance(document, list):  # a receipt chain: every receipt is signed
                signed: Any = [signer.sign(item) for item in document]
            elif isinstance(document, dict):
                signed = signer.sign(document)
            else:
                raise ValueError("expected a JSON object or list")
            target = Path(args.output or args.file)
            target.write_text(json.dumps(signed, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"dllm sign: {error}", file=sys.stderr)
        return 2
    print(f"signed:             {target} (key {signer.public_key})")
    return 0


def _signature_problems(document: Any, trust: list[str]) -> list[str] | None:
    """Why ``document`` (or each receipt of a chain) lacks a valid signature by a trusted key; ``None`` when a
    trusted key is unusable."""
    if not trust:
        return []
    from etalii_dllm import signing

    try:
        keys = [signing.read_public_key(key) for key in trust]
    except signing.SigningError as error:
        print(f"dllm replay: {error}", file=sys.stderr)
        return None
    items = document if isinstance(document, list) else [document]
    problems = []
    for index, item in enumerate(items):
        problem = signing.signature_problem(item, keys) if isinstance(item, dict) else "it is not a JSON object"
        if problem is not None:
            problems.append(f"turn {index}: {problem}" if isinstance(document, list) else problem)
    return problems


def _signed(code: int, problems: list[str]) -> int:
    """The replay's exit code, failed when a required signature is missing or invalid."""
    for problem in problems:
        print(f"signature:          {problem}", file=sys.stderr)
    if problems and code == 0:
        print("NOT verified: the signature check failed", file=sys.stderr)
        return 1
    return code


def _read_json(path: str) -> Any:
    return json.loads(sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8"))


def _replay_training(args: argparse.Namespace) -> int:
    from etalii_dllm.modelfile import ModelFileError
    from etalii_dllm.training import TrainingDataError
    from etalii_dllm.training.receipt import verify

    receipt = args.loaded
    base = args.base or args.model or os.environ.get(MODEL_ENVIRONMENT_VARIABLE)
    if not base:
        print("dllm replay: a training receipt needs the base model (--base or --model)", file=sys.stderr)
        return 2
    try:
        outcome = verify(receipt, base, args.data, teacher=args.teacher, prompts=args.prompts)
    except (OSError, ValueError, KeyError, ModelFileError, TrainingDataError) as error:
        print(f"dllm replay: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(outcome.to_json(), indent=2, ensure_ascii=False))
    else:
        print(f"training receipt:   {receipt.get('id')}  ({receipt['output']['steps']} steps)")
        print(f"base:               {receipt.get('base_fingerprint')}")
        for note in outcome.notes:
            print(f"note:               {note}")
        for reason in outcome.reasons:
            print(f"differs:            {reason}")
        verdict = "verified: training again gave the same weights, bit for bit"
        print(verdict if outcome.ok else "NOT verified")
    return 0 if outcome.ok else 1


def _replay(engine: DllmEngine, args: argparse.Namespace) -> int:
    from etalii_dllm import receipts

    try:
        receipt = args.loaded
        if isinstance(receipt, list):
            return _replay_chain(engine, receipt, args.json)
        if not isinstance(receipt, dict):
            raise ValueError(f"not a {receipts.FORMAT} receipt")
        if "transcript" in receipt:
            return _replay_transcript(engine, receipt, args.json)
        verification = receipts.verify(engine, receipt)
    except (OSError, ValueError, KeyError) as error:
        print(f"dllm replay: {error}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(verification.to_json(), indent=2, ensure_ascii=False))
    else:
        print(f"receipt:            {receipt.get('id')}")
        print(f"model:              {receipt.get('model')}  ({receipt.get('system_fingerprint')})")
        for note in verification.notes:
            print(f"note:               {note}")
        for reason in verification.reasons:
            print(f"differs:            {reason}")
        print("verified: the replay gave the same output, bit for bit" if verification.ok else "NOT verified")
    return 0 if verification.ok else 1


def _replay_chain(engine: DllmEngine, chain: list, as_json: bool) -> int:
    from etalii_dllm import receipts

    if not all(isinstance(r, dict) for r in chain):
        raise ValueError(f"a receipt chain is a list of {receipts.FORMAT} receipts")
    outcome = receipts.verify_chain(engine, chain)
    if as_json:
        print(json.dumps(outcome.to_json(), indent=2, ensure_ascii=False))
    else:
        print(f"chain:              {len(chain)} turns, {chain[0].get('id')} .. {chain[-1].get('id')}")
        print(f"model:              {chain[0].get('model')}  ({chain[0].get('system_fingerprint')})")
        for note in outcome.notes:
            print(f"note:               {note}")
        for reason in outcome.reasons:
            print(f"differs:            {reason}")
        verdict = "verified: every turn gave the same output and continues the one before"
        print(verdict if outcome.ok else "NOT verified")
    return 0 if outcome.ok else 1


def _replay_transcript(engine: DllmEngine, transcript: dict, as_json: bool) -> int:
    from etalii_dllm import transcripts

    outcome = transcripts.replay(engine, transcript)
    if as_json:
        print(json.dumps(outcome.to_json(), indent=2, ensure_ascii=False))
    else:
        print(f"transcript:         {transcript.get('id')}  ({len(transcript['rounds'])} rounds)")
        print(f"model:              {transcript.get('model')}  ({transcript.get('system_fingerprint')})")
        for note in outcome.notes:
            print(f"note:               {note}")
        for reason in outcome.reasons:
            print(f"differs:            {reason}")
        verdict = "verified: every round gave the same output, bit for bit"
        print(verdict if outcome.ok else "NOT verified")
    return 0 if outcome.ok else 1


def _write_receipt(path: str | None, finished: Finished) -> None:
    if path and finished.receipt is not None:
        Path(path).write_text(json.dumps(finished.receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"receipt: {path} ({finished.receipt['id']})", file=sys.stderr)


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
    if args.regex is not None:
        if args.json or args.json_schema:
            print("dllm chat: --regex cannot be combined with --json or --json-schema", file=sys.stderr)
            return 1
        response_format = ResponseFormat("regex", pattern=args.regex)
    messages = [ChatMessage("system", args.system)] if args.system else []
    request = ChatRequest(
        [*messages, ChatMessage("user", args.message)], args.max_tokens, options, response_format=response_format
    )
    if args.mcp_config or args.mcp_server or args.tool:
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
            _write_receipt(args.receipt, event)
    return 0


def _chat_with_mcp(engine: DllmEngine, args: argparse.Namespace, request: ChatRequest) -> int:
    import anyio

    from etalii_dllm import builtin_tools, mcp_host, transcripts
    from etalii_dllm.engine import ToolCallEvent

    async def run() -> int:
        finished: Finished | None = None
        configs = mcp_host.load_config(args.mcp_config) if args.mcp_config else []
        configs += [mcp_host.parse_server(spec) for spec in args.mcp_server]
        servers: list[mcp_host.McpServerConfig] | dict[str, Any] = configs
        if args.tool:
            names = [c.name for c in configs]
            if len(set(names)) != len(names) or "tools" in names:
                raise mcp_host.McpHostError("MCP server names must be unique ('tools' is the built-in tools)")
            servers = {**{c.name: c for c in configs}, "tools": builtin_tools.server(args.tool, engine)}
        async with mcp_host.McpHost(servers) as host:
            print(f"tools: {', '.join(t.name for t in host.tools) or '(none)'}", file=sys.stderr)
            recorder = None
            if args.transcript:
                recorder = transcripts.Recorder(engine, request, host.tools, args.max_tool_rounds, host.servers)
            async for event in mcp_host.chat(engine, request, host, args.max_tool_rounds):
                if recorder is not None:
                    recorder.add(event)
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
        _write_receipt(args.receipt, finished)  # the last round's request carries the tool results
        if recorder is not None:
            transcript = recorder.transcript()
            text = json.dumps(transcript, indent=2, ensure_ascii=False) + "\n"
            Path(args.transcript).write_text(text, encoding="utf-8")
            print(f"transcript: {args.transcript} ({transcript['id']})", file=sys.stderr)
        return 0

    try:
        return anyio.run(run)
    except (mcp_host.McpHostError, ValueError) as error:
        print(f"dllm chat: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
