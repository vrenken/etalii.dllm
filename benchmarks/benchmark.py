"""Model performance comparison: EtAlii.Dllm against transformers (PyTorch) and llama.cpp on the same machine.

The methodology follows the established comparisons (``docs/benchmarks.md``):

* throughput like llama.cpp's ``llama-bench``: prompt processing ``ppN`` and token generation ``tgN`` in tokens per
  second, the mean and standard deviation over repetitions after a warm-up run;
* latency like MLPerf Inference and vLLM's ``benchmark_serving``: time to first token (TTFT), time per output token
  (TPOT, with p50/p99) and aggregate throughput for 1, 4 and 8 concurrent requests;
* quality like llama.cpp's ``llama-perplexity``: perplexity on the WikiText-2 test set in 512-token chunks, scoring
  the second half of each chunk, and for Q8_0 the KL divergence and top-token agreement against float32;
* determinism like Thinking Machines' "Defeating Nondeterminism in LLM Inference": repeated runs, thread counts and
  batch composition must not change a single bit.

``python benchmarks/benchmark.py suite --snapshots DIR --output results.json`` runs everything; every measurement
runs in a child process of its own so peak memory (RSS) is per configuration. ``report`` renders the JSON as
Markdown tables. Benchmark code, not inference code: NumPy reductions are fine here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# WikiText-2 test split (the tokenised version shipped with pytorch/examples), pinned by commit and hash.
CORPUS_URL = "https://raw.githubusercontent.com/pytorch/examples/{commit}/word_language_model/data/wikitext-2/test.txt"
CORPUS_COMMIT = "main"
CORPUS_SHA256 = "d790b833ef8cf03a90db7bf1271b7520b83c45ce07ba3c1a9699df81e239eca0"

MODELS = {
    "SmolLM2-135M": ("HuggingFaceTB/SmolLM2-135M-Instruct", "12fd25f77366fa6b3b4b768ec3050bf629380bac"),
    "Qwen2.5-0.5B": ("Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"),
    "Qwen3-0.6B": ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"),
    "OLMo-2-1B": ("allenai/OLMo-2-0425-1B-Instruct", "48d788eca847d4d7548f375ad03d3c9312f6139e"),
    "TinyLlama-1.1B": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "fe8a4ea1ffedaf415f4da2f062534de366a451e6"),
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"),
}

CONCURRENT_PROMPTS = [
    "The history of the city begins in the",
    "In mathematics, a prime number is",
    "The recipe calls for two cups of flour and",
    "During the second world war , the",
    "The river flows from the mountains to",
    "A computer program is a sequence of",
    "The album was released in 1998 and",
    "Photosynthesis is the process by which",
]


# -- helpers ----------------------------------------------------------------------------------------------------


def peak_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if sys.platform == "darwin" else usage / 1024


def summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "n": len(values),
    }


def percentile(values: Sequence[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokens_hash(tokens: Sequence[int]) -> str:
    return sha(np.asarray(tokens, dtype=np.int64).tobytes())[:16]


def logits_hash(logits: Any) -> str:
    return sha(np.ascontiguousarray(logits, dtype=np.float32).tobytes())[:16]


def corpus(cache: Path) -> str:
    """The WikiText-2 test text: downloaded once, checked against the pinned hash when one is set."""
    path = cache / "wikitext-2-test.txt"
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(CORPUS_URL.format(commit=CORPUS_COMMIT)) as response:
            path.write_bytes(response.read())
    data = path.read_bytes()
    if CORPUS_SHA256 and sha(data) != CORPUS_SHA256:
        raise SystemExit(f"{path}: unexpected content (sha256 {sha(data)})")
    return data.decode("utf-8")


def machine() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cores": os.cpu_count(),
    }
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        found = re.search(r"model name\s*:\s*(.+)", cpuinfo)
        info["cpu"] = found.group(1).strip() if found else platform.processor()
        memory = re.search(r"MemTotal:\s*(\d+)", Path("/proc/meminfo").read_text(encoding="utf-8"))
        info["memory_gb"] = round(int(memory.group(1)) / 1024 / 1024, 1) if memory else None
    except OSError:
        info["cpu"] = platform.processor()
    try:
        from etalii_dllm import __version__
        from etalii_dllm.numerics import instruction_set

        info["dllm"] = __version__
        info["instruction_set"] = instruction_set()
    except ImportError:
        pass
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT, check=True)
        info["commit"] = commit.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return info


def child(arguments: list[str], timeout: float = 7200) -> dict[str, Any]:
    """Runs ``benchmark.py ARGUMENTS`` in a fresh process and returns the JSON it prints last."""
    print("  $", " ".join(arguments), file=sys.stderr, flush=True)
    process = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *arguments],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if process.returncode != 0:
        print(process.stdout[-4000:], process.stderr[-4000:], sep="\n", file=sys.stderr, flush=True)
        return {"error": (process.stderr.strip().splitlines() or ["failed"])[-1]}
    return json.loads(process.stdout.strip().splitlines()[-1])


# -- EtAlii.Dllm ------------------------------------------------------------------------------------------------


def run_dllm(args: argparse.Namespace) -> dict[str, Any]:
    from etalii_dllm.engine import DllmEngine
    from etalii_dllm.numerics import argmax, linear, log_softmax, set_threads, threads
    from etalii_dllm.sampling import GREEDY

    quantize = None if args.quantize == "none" else args.quantize
    set_threads(args.threads)
    started = time.perf_counter()
    engine = DllmEngine.from_model_file(args.model, quantize=quantize, device=args.device, prompt_cache=0)
    load_s = time.perf_counter() - started
    model = engine.model
    text = corpus(Path(args.cache))
    tokens = engine.tokenizer.encode(text[: args.corpus_chars])
    result: dict[str, Any] = {
        "engine": "dllm",
        "quantize": args.quantize,
        "device": args.device,
        "threads": threads(),
        "load_s": load_s,
        "fingerprint": engine.system_fingerprint,
    }

    def prefill(count: int) -> tuple[float, str]:
        cache = model.new_cache()
        begin = time.perf_counter()
        logits = model.forward_cached(tokens[:count], cache)
        return count / (time.perf_counter() - begin), logits_hash(logits)

    def decode(count: int, prompt: int = 16) -> tuple[list[float], list[int]]:
        cache = model.new_cache()
        sequence = list(tokens[:prompt])
        logits = model.forward_cached(sequence, cache)
        steps = []
        for _ in range(count):
            sequence.append(argmax(logits))
            begin = time.perf_counter()
            logits = model.forward_cached(sequence, cache)
            steps.append(time.perf_counter() - begin)
        return steps, sequence[prompt:]

    # Throughput (llama-bench): one warm-up, then the repetitions.
    prefill(args.pp)
    pp = [prefill(args.pp) for _ in range(args.reps)]
    decode(4)
    tg = [decode(args.tg) for _ in range(args.reps)]
    step_times = [step for steps, _ in tg for step in steps]
    result["pp"] = {"tokens": args.pp, **summary([rate for rate, _ in pp])}
    result["tg"] = {"tokens": args.tg, **summary([args.tg / sum(steps) for steps, _ in tg])}
    result["tpot_ms"] = {
        "p50": percentile(step_times, 50) * 1000,
        "p99": percentile(step_times, 99) * 1000,
        "mean": statistics.fmean(step_times) * 1000,
    }
    result["determinism"] = {
        "repeat_prefill_logits_identical": len({digest for _, digest in pp}) == 1,
        "repeat_decode_tokens_identical": len({tokens_hash(out) for _, out in tg}) == 1,
        "prefill_logits_hash": pp[0][1],
        "decode_tokens_hash": tokens_hash(tg[0][1]),
    }

    # Thread scaling: speed may change, the bits may not.
    sweep = []
    for count in args.thread_sweep:
        set_threads(count)
        rate, digest = prefill(args.pp)
        steps, out = decode(min(args.tg, 32))
        sweep.append(
            {
                "threads": count,
                "pp_tps": rate,
                "tg_tps": len(steps) / sum(steps),
                "prefill_logits_hash": digest,
                "decode_tokens_hash": tokens_hash(out),
            }
        )
    set_threads(args.threads)
    result["thread_sweep"] = sweep
    result["determinism"]["threads_identical"] = (
        len({row["prefill_logits_hash"] for row in sweep}) == 1
        and len({row["decode_tokens_hash"] for row in sweep}) == 1
        and sweep[0]["prefill_logits_hash"] == result["determinism"]["prefill_logits_hash"]
    )

    # Batch invariance: a prompt alone versus stacked with prompts of other lengths in one forward pass.
    probe = tokens[:48]
    others = [tokens[100:117], tokens[300:391], tokens[500:503]]
    alone = model.forward_cached(probe, model.new_cache())
    batched = model.forward_batch([others[0], probe, *others[1:]])[1]
    result["determinism"]["batch_invariant"] = bool(np.array_equal(alone, batched))
    result["determinism"]["batch_max_abs_diff"] = float(np.max(np.abs(alone - batched)))

    # Concurrency (vLLM benchmark_serving style): N requests at once share batched decode steps.
    solo: dict[str, tuple[int, ...]] = {}
    concurrency = []
    for users in args.concurrency:
        prompts = CONCURRENT_PROMPTS[:users]
        ttft: list[float] = [0.0] * users
        outputs: list[tuple[int, ...]] = [()] * users
        barrier = threading.Barrier(users)

        def request(i: int, prompts: list[str] = prompts, ttft: list[float] = ttft, outputs=outputs, b=barrier):
            b.wait()
            begin = time.perf_counter()
            generation = engine.complete_stream(prompts[i], args.concurrent_tokens, GREEDY)
            for step in generation:
                if step.token is not None and not ttft[i]:
                    ttft[i] = time.perf_counter() - begin
            outputs[i] = generation.result().tokens

        workers = [threading.Thread(target=request, args=(i,)) for i in range(users)]
        begin = time.perf_counter()
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join()
        elapsed = time.perf_counter() - begin
        generated = sum(len(out) for out in outputs)
        concurrency.append(
            {
                "users": users,
                "elapsed_s": elapsed,
                "output_tps": generated / elapsed,
                "ttft_ms_p50": percentile(ttft, 50) * 1000,
                "ttft_ms_p99": percentile(ttft, 99) * 1000,
                "outputs": [tokens_hash(out) for out in outputs],
            }
        )
    for prompt in CONCURRENT_PROMPTS[: max(args.concurrency)]:
        solo[prompt] = engine.complete(prompt, args.concurrent_tokens, GREEDY).tokens
    identical = all(
        row["outputs"][i] == tokens_hash(solo[CONCURRENT_PROMPTS[i]])
        for row in concurrency
        for i in range(row["users"])
    )
    result["concurrency"] = concurrency
    result["determinism"]["concurrent_equals_solo"] = identical

    # Perplexity (llama-perplexity): chunks of ctx tokens, score the second half of each.
    nll: list[float] = []
    top1 = 0
    kl: list[float] = []
    reference = Path(args.kld_reference) if args.kld_reference else None
    begin = time.perf_counter()
    first = args.ctx // 2
    for chunk in range(args.ppl_chunks):
        window = tokens[chunk * args.ctx : (chunk + 1) * args.ctx]
        if len(window) < args.ctx:
            break
        hidden = model.hidden_states(window)
        logits = linear(np.ascontiguousarray(hidden[first : args.ctx - 1]), model._lm_head).numpy()
        logprobs = np.stack([log_softmax(row) for row in logits])
        targets = np.asarray(window[first + 1 :])
        nll.extend((-logprobs[np.arange(len(targets)), targets]).tolist())
        if reference is not None and chunk < args.kld_chunks:
            path = reference / f"chunk{chunk}.npy"
            if quantize is None:
                reference.mkdir(parents=True, exist_ok=True)
                np.save(path, logprobs)
            elif path.exists():
                base = np.load(path)
                kl.extend(np.sum(np.exp(base) * (base - logprobs), axis=1).tolist())
                top1 += int(np.sum(np.argmax(base, axis=1) == np.argmax(logprobs, axis=1)))
    scored = len(nll)
    mean = statistics.fmean(nll)
    result["perplexity"] = {
        "ppl": float(np.exp(mean)),
        "ppl_stderr": float(np.exp(mean) * statistics.stdev(nll) / np.sqrt(scored)),
        "scored_tokens": scored,
        "ctx": args.ctx,
        "eval_tps": (scored * 2) / (time.perf_counter() - begin),
    }
    if kl:
        result["perplexity"]["kld_mean"] = statistics.fmean(kl)
        result["perplexity"]["kld_p99"] = percentile(kl, 99)
        result["perplexity"]["top1_agreement"] = top1 / len(kl)
    result["corpus_tokens"] = len(tokens)
    result["peak_rss_mb"] = peak_rss_mb()
    return result


# -- transformers (PyTorch) -------------------------------------------------------------------------------------


def run_transformers(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import transformers

    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model.eval()
    load_s = time.perf_counter() - started
    tokens = tokenizer.encode(corpus(Path(args.cache))[: args.corpus_chars], add_special_tokens=False)
    result: dict[str, Any] = {
        "engine": "transformers",
        "version": f"transformers {transformers.__version__}, torch {torch.__version__}",
        "quantize": "none",
        "threads": torch.get_num_threads(),
        "load_s": load_s,
    }

    @torch.inference_mode()
    def prefill(count: int) -> tuple[float, str]:
        begin = time.perf_counter()
        logits = model(torch.tensor([tokens[:count]])).logits[0, -1]
        return count / (time.perf_counter() - begin), logits_hash(logits.numpy())

    @torch.inference_mode()
    def decode(count: int, prompt: int = 16) -> tuple[list[float], list[int]]:
        output = model(torch.tensor([tokens[:prompt]]), use_cache=True)
        past, logits = output.past_key_values, output.logits[0, -1]
        steps, out = [], []
        for _ in range(count):
            token = int(torch.argmax(logits))
            out.append(token)
            begin = time.perf_counter()
            output = model(torch.tensor([[token]]), past_key_values=past, use_cache=True)
            past, logits = output.past_key_values, output.logits[0, -1]
            steps.append(time.perf_counter() - begin)
        return steps, out

    prefill(args.pp)
    pp = [prefill(args.pp) for _ in range(args.reps)]
    decode(4)
    tg = [decode(args.tg) for _ in range(args.reps)]
    step_times = [step for steps, _ in tg for step in steps]
    result["pp"] = {"tokens": args.pp, **summary([rate for rate, _ in pp])}
    result["tg"] = {"tokens": args.tg, **summary([args.tg / sum(steps) for steps, _ in tg])}
    result["tpot_ms"] = {
        "p50": percentile(step_times, 50) * 1000,
        "p99": percentile(step_times, 99) * 1000,
        "mean": statistics.fmean(step_times) * 1000,
    }
    result["determinism"] = {
        "repeat_prefill_logits_identical": len({digest for _, digest in pp}) == 1,
        "repeat_decode_tokens_identical": len({tuple(out) for _, out in tg}) == 1,
        "prefill_logits_hash": pp[0][1],
        "decode_tokens_hash": tokens_hash(tg[0][1]),
    }
    sweep = []
    for count in args.thread_sweep:
        torch.set_num_threads(count)
        rate, digest = prefill(args.pp)
        steps, out = decode(min(args.tg, 32))
        sweep.append(
            {
                "threads": count,
                "pp_tps": rate,
                "tg_tps": len(steps) / sum(steps),
                "prefill_logits_hash": digest,
                "decode_tokens_hash": tokens_hash(out),
            }
        )
    torch.set_num_threads(args.threads)
    result["thread_sweep"] = sweep
    result["determinism"]["threads_identical"] = (
        len({row["prefill_logits_hash"] for row in sweep}) == 1
        and len({row["decode_tokens_hash"] for row in sweep}) == 1
    )

    # Batch invariance: the same prompt alone and left-padded in a batch with prompts of other lengths.
    with torch.inference_mode():
        probe = tokens[:48]
        others = [tokens[100:117], tokens[300:391], tokens[500:503]]
        batch = [others[0], probe, *others[1:]]
        width = max(len(row) for row in batch)
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        ids = torch.tensor([[pad] * (width - len(row)) + row for row in batch])
        mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in batch])
        alone = model(torch.tensor([probe])).logits[0, -1].numpy()
        batched = model(ids, attention_mask=mask).logits[1, -1].numpy()
    result["determinism"]["batch_invariant"] = bool(np.array_equal(alone, batched))
    result["determinism"]["batch_max_abs_diff"] = float(np.max(np.abs(alone - batched)))
    result["determinism"]["batch_same_top_token"] = int(np.argmax(alone)) == int(np.argmax(batched))

    nll: list[float] = []
    first = args.ctx // 2
    begin = time.perf_counter()
    with torch.inference_mode():
        for chunk in range(args.ppl_chunks):
            window = tokens[chunk * args.ctx : (chunk + 1) * args.ctx]
            if len(window) < args.ctx:
                break
            logits = model(torch.tensor([window])).logits[0, first : args.ctx - 1]
            logprobs = torch.log_softmax(logits.double(), dim=-1)
            targets = torch.tensor(window[first + 1 :])
            nll.extend((-logprobs[torch.arange(len(targets)), targets]).tolist())
    mean = statistics.fmean(nll)
    result["perplexity"] = {
        "ppl": float(np.exp(mean)),
        "ppl_stderr": float(np.exp(mean) * statistics.stdev(nll) / np.sqrt(len(nll))),
        "scored_tokens": len(nll),
        "ctx": args.ctx,
        "eval_tps": (len(nll) * 2) / (time.perf_counter() - begin),
    }
    result["corpus_tokens"] = len(tokens)
    result["peak_rss_mb"] = peak_rss_mb()
    return result


# -- llama.cpp --------------------------------------------------------------------------------------------------


def llamacpp_determinism(args: argparse.Namespace) -> dict[str, Any] | None:
    """The same probes as for the other engines, through the llama-cpp-python bindings (they give the raw logits):
    repeated runs, thread counts, and the micro-batch size, llama.cpp's analogue of batch composition."""
    try:
        import llama_cpp
    except ImportError:
        return None
    text = corpus(Path(args.cache))[: args.corpus_chars].encode("utf-8")

    def run(threads: int, ubatch: int) -> tuple[np.ndarray, list[int]]:
        llm = llama_cpp.Llama(
            args.model,
            n_ctx=512,
            n_threads=threads,
            n_threads_batch=threads,
            n_batch=512,
            n_ubatch=ubatch,
            logits_all=True,
            verbose=False,
        )
        tokens = llm.tokenize(text, add_bos=False)[: args.pp]
        llm.eval(tokens)
        logits = np.array(llm.scores[: len(tokens)], dtype=np.float32)
        generated = []
        for _ in range(min(args.tg, 32)):
            token = int(np.argmax(llm.scores[llm.n_tokens - 1]))
            generated.append(token)
            llm.eval([token])
        return logits, generated

    base, base_tokens = run(args.threads, 512)
    again, again_tokens = run(args.threads, 512)
    single, single_tokens = run(1, 512)
    small, small_tokens = run(args.threads, 16)
    return {
        "version": f"llama-cpp-python {llama_cpp.__version__}",
        "repeat_prefill_logits_identical": bool(np.array_equal(base, again)),
        "repeat_decode_tokens_identical": base_tokens == again_tokens,
        "threads_identical": bool(np.array_equal(base, single)) and base_tokens == single_tokens,
        "threads_max_abs_diff": float(np.max(np.abs(base - single))),
        "batch_invariant": bool(np.array_equal(base, small)) and base_tokens == small_tokens,
        "batch_max_abs_diff": float(np.max(np.abs(base - small))),
        "batch_same_top_token": bool(np.array_equal(np.argmax(base, axis=1), np.argmax(small, axis=1))),
    }


def llamacpp_server(args: argparse.Namespace, binary: Path) -> tuple[list[dict[str, Any]], bool, bool]:
    """llama.cpp's own continuous batching: ``llama-server`` with parallel slots, greedy requests streamed, first
    one at a time and then N at once. Compares the text and the chosen tokens' log-probabilities with the solo runs,
    and measures the same throughput and TTFT as for EtAlii.Dllm."""
    port = 8089
    slots = max(args.concurrency)
    command = [str(binary), "-m", args.model, "-np", str(slots), "-c", str(512 * slots), "-t", str(args.threads)]
    process = subprocess.Popen(
        [*command, "--port", str(port), "--no-webui"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(600):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health") as response:
                    if response.status == 200:
                        break
            except OSError:
                time.sleep(0.5)

        def complete(prompt: str) -> tuple[float, float, int, str, str]:
            body = {
                "prompt": prompt,
                "n_predict": args.concurrent_tokens,
                "temperature": 0,
                "top_k": 1,
                "cache_prompt": False,
                "n_probs": 1,
                "stream": True,
            }
            request = urllib.request.Request(
                f"http://127.0.0.1:{port}/completion",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            )
            begin = time.perf_counter()
            first = 0.0
            pieces: list[Any] = []
            count = 0
            with urllib.request.urlopen(request) as response:
                for raw in response:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    if event.get("content") and not first:
                        first = time.perf_counter() - begin
                    if event.get("content"):
                        count += 1
                    pieces.append([event.get("content"), event.get("completion_probabilities")])
            text = "".join(piece[0] or "" for piece in pieces)
            elapsed = time.perf_counter() - begin
            return first, elapsed, count, sha(json.dumps(pieces).encode())[:16], sha(text.encode())[:16]

        solo = {prompt: complete(prompt) for prompt in CONCURRENT_PROMPTS[:slots]}
        rows = []
        identical = True
        same_text = True
        for users in args.concurrency:
            prompts = CONCURRENT_PROMPTS[:users]
            answers: list[tuple[float, float, int, str, str]] = [(0.0, 0.0, 0, "", "")] * users
            barrier = threading.Barrier(users)

            def request(i: int, prompts: list[str] = prompts, answers=answers, b=barrier) -> None:
                b.wait()
                answers[i] = complete(prompts[i])

            workers = [threading.Thread(target=request, args=(i,)) for i in range(users)]
            begin = time.perf_counter()
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join()
            elapsed = time.perf_counter() - begin
            identical = identical and all(answers[i][3] == solo[prompts[i]][3] for i in range(users))
            same_text = same_text and all(answers[i][4] == solo[prompts[i]][4] for i in range(users))
            ttft = [answer[0] for answer in answers]
            rows.append(
                {
                    "users": users,
                    "elapsed_s": elapsed,
                    "output_tps": sum(answer[2] for answer in answers) / elapsed,
                    "ttft_ms_p50": percentile(ttft, 50) * 1000,
                    "ttft_ms_p99": percentile(ttft, 99) * 1000,
                    "outputs": [answer[3] for answer in answers],
                }
            )
        return rows, identical, same_text
    finally:
        process.terminate()
        process.wait(timeout=60)


def run_llamacpp(args: argparse.Namespace) -> dict[str, Any]:
    """Runs the llama.cpp tools on a GGUF file: ``llama-bench`` for throughput, ``llama-perplexity`` for quality."""
    binaries = Path(args.llamacpp)
    bench = subprocess.run(
        [
            str(binaries / "llama-bench"),
            "-m",
            args.model,
            "-p",
            str(args.pp),
            "-n",
            str(args.tg),
            "-r",
            str(args.reps),
            "-t",
            str(args.threads),
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(bench.stdout)
    result: dict[str, Any] = {"engine": "llama.cpp", "quantize": args.quantize, "threads": args.threads}
    for row in rows:
        key = "pp" if row.get("n_prompt", 0) > 0 else "tg"
        count = row["n_prompt"] if key == "pp" else row["n_gen"]
        result[key] = {"tokens": count, "mean": row["avg_ts"], "stdev": row["stddev_ts"], "n": args.reps}
        result["version"] = f"llama.cpp {row.get('build_commit', '')} (build {row.get('build_number', '')})"
    sweep = []
    for count in args.thread_sweep:
        rerun = subprocess.run(
            [
                str(binaries / "llama-bench"),
                "-m",
                args.model,
                "-p",
                str(args.pp),
                "-n",
                str(min(args.tg, 32)),
                "-r",
                "1",
                "-t",
                str(count),
                "-o",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        speeds = {("pp" if r.get("n_prompt", 0) > 0 else "tg"): r["avg_ts"] for r in json.loads(rerun.stdout)}
        sweep.append({"threads": count, "pp_tps": speeds.get("pp"), "tg_tps": speeds.get("tg")})
    result["thread_sweep"] = sweep
    result["determinism"] = llamacpp_determinism(args)
    server = binaries / "llama-server"
    if server.exists():
        result["concurrency"], solo_equal, same_text = llamacpp_server(args, server)
        if result["determinism"] is not None:
            result["determinism"]["concurrent_equals_solo"] = solo_equal
            result["determinism"]["concurrent_same_text"] = same_text
    text = Path(args.cache) / "wikitext-2-test.txt"
    corpus(Path(args.cache))
    perplexity = subprocess.run(
        [
            str(binaries / "llama-perplexity"),
            "-m",
            args.model,
            "-f",
            str(text),
            "-c",
            str(args.ctx),
            "--chunks",
            str(args.ppl_chunks),
            "-t",
            str(args.threads),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    found = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", perplexity.stdout + perplexity.stderr)
    if found:
        result["perplexity"] = {"ppl": float(found.group(1)), "ppl_stderr": float(found.group(2)), "ctx": args.ctx}
    return result


# -- suite ------------------------------------------------------------------------------------------------------


def common_arguments(args: argparse.Namespace) -> list[str]:
    return [
        "--threads",
        str(args.threads),
        "--reps",
        str(args.reps),
        "--pp",
        str(args.pp),
        "--tg",
        str(args.tg),
        "--ctx",
        str(args.ctx),
        "--ppl-chunks",
        str(args.ppl_chunks),
        "--corpus-chars",
        str(args.corpus_chars),
        "--cache",
        str(args.cache),
        "--thread-sweep",
        *map(str, args.thread_sweep),
    ]


def suite(args: argparse.Namespace) -> dict[str, Any]:
    from etalii_dllm.importing import import_model

    results: dict[str, Any] = {"machine": machine(), "settings": vars(args).copy(), "models": {}}
    results["settings"].pop("func", None)
    corpus(Path(args.cache))
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)

    def save() -> None:
        output.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")

    for name in args.models:
        repository, revision = MODELS[name]
        snapshot = Path(args.snapshots) / repository / revision
        if not snapshot.exists():
            candidates = sorted((Path(args.snapshots) / repository).glob("*/"))
            snapshot = candidates[0] if candidates else Path(args.snapshots) / repository.split("/")[1]
        print(f"== {name} ({snapshot})", file=sys.stderr, flush=True)
        entry: dict[str, Any] = {"repository": repository, "revision": revision, "runs": []}
        results["models"][name] = entry
        dllm_file = work / f"{name}.dllm"
        begin = time.perf_counter()
        import_model(snapshot, dllm_file, repository=repository, revision=revision, licence="Apache-2.0")
        entry["import_s"] = time.perf_counter() - begin
        entry["dllm_file_mb"] = dllm_file.stat().st_size / 2**20
        kld = work / f"{name}-kld"
        for quantize in ("none", "q8_0"):
            run = child(
                [
                    "dllm",
                    "--model",
                    str(dllm_file),
                    "--quantize",
                    quantize,
                    "--kld-reference",
                    str(kld),
                    "--kld-chunks",
                    str(args.kld_chunks),
                    "--concurrency",
                    *map(str, args.concurrency),
                    "--concurrent-tokens",
                    str(args.concurrent_tokens),
                    *common_arguments(args),
                ]
            )
            entry["runs"].append(run)
            save()
        shutil.rmtree(kld, ignore_errors=True)
        dllm_file.unlink()
        if not args.skip_transformers:
            entry["runs"].append(child(["transformers", "--model", str(snapshot), *common_arguments(args)]))
            save()
        if args.llamacpp:
            for quantize, outtype in (("none", "f32"), ("q8_0", "q8_0")):
                gguf = work / f"{name}-{outtype}.gguf"
                converted = subprocess.run(
                    [
                        sys.executable,
                        str(Path(args.llamacpp_source) / "convert_hf_to_gguf.py"),
                        str(snapshot),
                        "--outfile",
                        str(gguf),
                        "--outtype",
                        outtype,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if converted.returncode != 0:
                    failure = converted.stderr[-500:]
                    entry["runs"].append({"engine": "llama.cpp", "quantize": quantize, "error": failure})
                    continue
                entry[f"gguf_{outtype}_mb"] = gguf.stat().st_size / 2**20
                entry["runs"].append(
                    child(
                        [
                            "llamacpp",
                            "--model",
                            str(gguf),
                            "--quantize",
                            quantize,
                            "--llamacpp",
                            args.llamacpp,
                            "--concurrency",
                            *map(str, args.concurrency),
                            "--concurrent-tokens",
                            str(args.concurrent_tokens),
                            *common_arguments(args),
                        ]
                    )
                )
                gguf.unlink()
                save()
        if args.delete_snapshots:
            shutil.rmtree(snapshot, ignore_errors=True)
    save()
    return results


# -- report -----------------------------------------------------------------------------------------------------


def fmt(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:,.{digits}f}"


def report(results: dict[str, Any]) -> str:
    lines = []
    m = results["machine"]
    lines.append(
        f"Machine: {m.get('cpu')}, {m.get('cores')} cores, {m.get('memory_gb')} GB, {m.get('platform')}; "
        f"dllm {m.get('dllm')} ({m.get('instruction_set')}), commit {str(m.get('commit', ''))[:10]}."
    )
    s = results["settings"]
    lines.append("")
    lines.append(f"### Throughput (pp{s['pp']}, tg{s['tg']}, {s['threads']} threads, mean ± stdev of {s['reps']})")
    lines.append("")
    lines.append("| Model | Engine | Weights | pp t/s | tg t/s | TPOT p50 / p99 ms | Load s | Peak RSS MB |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    def label(run: dict[str, Any]) -> str:
        return {"none": "f32", "q8_0": "Q8_0"}.get(run.get("quantize", ""), str(run.get("quantize")))

    for name, entry in results["models"].items():
        for run in entry["runs"]:
            if "error" in run:
                lines.append(f"| {name} | {run.get('engine', '?')} | {label(run)} | error | | | | |")
                continue
            pp, tg, tpot = run.get("pp", {}), run.get("tg", {}), run.get("tpot_ms")
            lines.append(
                f"| {name} | {run['engine']} | {label(run)} | {fmt(pp.get('mean'))} ± {fmt(pp.get('stdev'))} | "
                f"{fmt(tg.get('mean'))} ± {fmt(tg.get('stdev'))} | "
                f"{(fmt(tpot['p50']) + ' / ' + fmt(tpot['p99'])) if tpot else 'n/a'} | "
                f"{fmt(run.get('load_s'), 2)} | {fmt(run.get('peak_rss_mb'), 0)} |"
            )
    lines.append("")
    lines.append(f"### Quality (WikiText-2 test, ctx {s['ctx']}, {s['ppl_chunks']} chunks)")
    lines.append("")
    lines.append("| Model | Engine | Weights | PPL | KLD vs f32 | Top-1 agreement |")
    lines.append("|---|---|---|---:|---:|---:|")
    for name, entry in results["models"].items():
        for run in entry["runs"]:
            p = run.get("perplexity")
            if not p:
                continue
            agreement = p.get("top1_agreement")
            lines.append(
                f"| {name} | {run['engine']} | {label(run)} | {fmt(p['ppl'], 3)} ± {fmt(p.get('ppl_stderr'), 3)} | "
                f"{fmt(p.get('kld_mean'), 5) if 'kld_mean' in p else ''} | "
                f"{fmt(agreement * 100, 2) + ' %' if agreement is not None else ''} |"
            )
    lines.append("")
    lines.append("### Determinism")
    lines.append("")
    lines.append("| Model | Engine | Weights | Repeat runs | Thread counts | Batch composition | Concurrent = solo |")
    lines.append("|---|---|---|---|---|---|---|")

    def mark(value: Any) -> str:
        return "n/a" if value is None else ("identical" if value else "**differs**")

    for name, entry in results["models"].items():
        for run in entry["runs"]:
            d = run.get("determinism")
            if not d:
                continue
            threads = mark(d.get("threads_identical"))
            if d.get("threads_identical") is False and "threads_max_abs_diff" in d:
                threads += f" (max {d['threads_max_abs_diff']:.2e})"
            concurrent = mark(d.get("concurrent_equals_solo"))
            if d.get("concurrent_equals_solo") is False and "concurrent_same_text" in d:
                concurrent += " (text " + ("same" if d["concurrent_same_text"] else "**differs**") + ")"
            batch = mark(d.get("batch_invariant"))
            if d.get("batch_invariant") is False:
                batch += f" (max {d['batch_max_abs_diff']:.2e})"
            lines.append(
                f"| {name} | {run['engine']} | {label(run)} | "
                f"{mark(d.get('repeat_prefill_logits_identical') and d.get('repeat_decode_tokens_identical'))} | "
                f"{threads} | {batch} | {concurrent} |"
            )
    lines.append("")
    lines.append("### Thread scaling (tokens/s)")
    lines.append("")
    counts = s["thread_sweep"]
    lines.append("| Model | Engine | Weights | " + " | ".join(f"pp @{c} / tg @{c}" for c in counts) + " |")
    lines.append("|---|---|---|" + "---:|" * len(counts))
    for name, entry in results["models"].items():
        for run in entry["runs"]:
            sweep = {row["threads"]: row for row in run.get("thread_sweep", [])}
            if not sweep:
                continue
            cells = [f"{fmt(sweep[c]['pp_tps'])} / {fmt(sweep[c]['tg_tps'])}" if c in sweep else "n/a" for c in counts]
            lines.append(f"| {name} | {run['engine']} | {label(run)} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(f"### Concurrent greedy requests ({s['concurrent_tokens']} tokens each)")
    lines.append("")
    lines.append("| Model | Engine | Weights | Users | Output t/s | TTFT p50 / p99 ms |")
    lines.append("|---|---|---|---:|---:|---:|")
    for name, entry in results["models"].items():
        for run in entry["runs"]:
            for row in run.get("concurrency", []):
                lines.append(
                    f"| {name} | {run['engine']} | {label(run)} | {row['users']} | {fmt(row['output_tps'])} | "
                    f"{fmt(row['ttft_ms_p50'], 0)} / {fmt(row['ttft_ms_p99'], 0)} |"
                )
    return "\n".join(lines) + "\n"


# -- command line -----------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def measured(command: argparse.ArgumentParser) -> None:
        command.add_argument("--threads", type=int, default=os.cpu_count() or 1)
        command.add_argument("--reps", type=int, default=5)
        command.add_argument("--pp", type=int, default=128)
        command.add_argument("--tg", type=int, default=64)
        command.add_argument("--ctx", type=int, default=512)
        command.add_argument("--ppl-chunks", type=int, default=8)
        command.add_argument("--corpus-chars", type=int, default=60_000)
        command.add_argument("--cache", default=str(Path(tempfile.gettempdir()) / "dllm-benchmark"))
        command.add_argument("--thread-sweep", type=int, nargs="+", default=[1, 2, 4])

    runs: dict[str, Callable[[argparse.Namespace], dict[str, Any]]] = {
        "dllm": run_dllm,
        "transformers": run_transformers,
        "llamacpp": run_llamacpp,
    }
    for name, function in runs.items():
        command = commands.add_parser(name)
        command.add_argument("--model", required=True)
        command.add_argument("--quantize", default="none")
        command.add_argument("--device", default="cpu")
        command.add_argument("--kld-reference")
        command.add_argument("--kld-chunks", type=int, default=4)
        command.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
        command.add_argument("--concurrent-tokens", type=int, default=32)
        command.add_argument("--llamacpp", help="directory with the llama-bench and llama-perplexity binaries")
        measured(command)
        command.set_defaults(func=function)
    whole = commands.add_parser("suite")
    whole.add_argument("--snapshots", required=True, help="hub cache with the pinned model snapshots")
    whole.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    whole.add_argument("--output", default="benchmark-results.json")
    whole.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "dllm-benchmark-work"))
    whole.add_argument("--kld-chunks", type=int, default=4)
    whole.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    whole.add_argument("--concurrent-tokens", type=int, default=32)
    whole.add_argument("--skip-transformers", action="store_true")
    whole.add_argument("--llamacpp", help="directory with the llama.cpp binaries (llama-bench, llama-perplexity)")
    whole.add_argument("--llamacpp-source", help="llama.cpp checkout (for convert_hf_to_gguf.py)")
    whole.add_argument("--delete-snapshots", action="store_true")
    measured(whole)
    whole.set_defaults(func=suite)
    render = commands.add_parser("report")
    render.add_argument("results")
    render.set_defaults(func=None)
    args = parser.parse_args(argv)
    if args.command == "report":
        print(report(json.loads(Path(args.results).read_text(encoding="utf-8"))), end="")
        return 0
    result = args.func(args)
    if args.command == "suite":
        print(report(result), end="")
    else:
        print(json.dumps(result, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
