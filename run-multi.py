#!/usr/bin/env python3
"""Utility for running BFCL multi-turn evaluations with OpenRouter-backed models.

This script orchestrates both generation and evaluation steps for the
`multi_turn_base` category, stores all artifacts under a dedicated report
folder, and writes helpful summaries so you can quickly compare the FC and
prompt variants of the same model.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from bfcl_eval.constants.model_config import api_inference_model_map

DEFAULT_CATEGORY = "multi_turn_base"
DATA_VERSION = "BFCL_v4"
MINI_TEST_IDS = [
    "multi_turn_base_0",
    "multi_turn_base_1",
    "multi_turn_base_2",
    "multi_turn_base_3",
    "multi_turn_base_4",
]


@dataclass
class RunSummary:
    mode: str
    model_id: str
    category: str
    mini: bool
    score: Dict[str, float]
    result_file: Path
    score_file: Path
    failures_file: Path | None = None

    def to_json(self) -> Dict[str, object]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mode": self.mode,
            "model_id": self.model_id,
            "category": self.category,
            "mini": self.mini,
            "score": self.score,
            "result_file": str(self.result_file),
            "score_file": str(self.score_file),
            "failures_file": str(self.failures_file) if self.failures_file else None,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Base model slug (e.g. anthropic/claude-haiku-4.5)",
    )
    parser.add_argument(
        "-d",
        "--dest",
        required=True,
        help="Directory where all reports, logs, and BFCL artifacts are stored",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Run only a minimal subset of multi_turn_base (useful for smoke tests)",
    )
    parser.add_argument(
        "--mode",
        choices=["fc", "prompt", "both"],
        default="both",
        help="Run only the FC variant, only the prompt variant, or both (default)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="Number of parallel threads to hand to bfcl generate (default: 1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature passed to bfcl generate",
    )
    parser.add_argument(
        "--http-referer",
        default="https://github.com/ShishirPatil/gorilla",
        help="Value for the HTTP-Referer header required by OpenRouter",
    )
    parser.add_argument(
        "--site-title",
        default="gorilla-bfcl",
        help="Value for the X-Title header required by OpenRouter",
    )
    parser.add_argument(
        "--category",
        default=DEFAULT_CATEGORY,
        help="Override the multi-turn category (defaults to multi_turn_base)",
    )
    return parser.parse_args()


def ensure_openrouter_env(args: argparse.Namespace) -> Dict[str, str]:
    env = os.environ.copy()
    key = env.get("OPENROUTER_API_KEY") or env.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY (or OPENAI_API_KEY) must be present in the environment"
        )

    env.setdefault("OPENAI_API_KEY", key)
    env.setdefault("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")

    default_headers = {
        "HTTP-Referer": args.http_referer,
        "X-Title": args.site_title,
    }
    if headers_env := env.get("OPENAI_DEFAULT_HEADERS"):
        try:
            existing = json.loads(headers_env)
        except json.JSONDecodeError:
            existing = {}
        default_headers |= existing
    env["OPENAI_DEFAULT_HEADERS"] = json.dumps(default_headers)

    return env


def resolve_model_ids(base_model: str, mode: str) -> List[Tuple[str, str]]:
    """Return (mode_label, bfcl_model_id) pairs to run."""

    options: List[Tuple[str, str]] = []
    if mode in ("fc", "both"):
        fc_id = f"{base_model}-FC"
        if fc_id not in api_inference_model_map:
            raise SystemExit(f"FC model '{fc_id}' is not registered in bfcl")
        options.append(("FC", fc_id))

    if mode in ("prompt", "both"):
        if base_model not in api_inference_model_map:
            raise SystemExit(f"Prompt model '{base_model}' is not registered in bfcl")
        options.append(("Prompt", base_model))

    return options


def write_subset_file(dest: Path, category: str, ids: Iterable[str]) -> Path:
    path = dest / "test_case_ids_to_generate.json"
    payload = {category: list(ids)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def run_command(cmd: List[str], env: Dict[str, str], log_path: Path, cwd: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed, see log: {log_path}")


def bfcl_paths(root: Path, model_id: str, category: str) -> Tuple[Path, Path]:
    model_folder = model_id.replace("/", "_")
    result_file = (
        root
        / "result"
        / model_folder
        / "multi_turn"
        / f"{DATA_VERSION}_{category}_result.json"
    )
    score_file = (
        root
        / "score"
        / model_folder
        / "multi_turn"
        / f"{DATA_VERSION}_{category}_score.json"
    )
    return result_file, score_file


def load_score(score_file: Path) -> Dict[str, float]:
    if not score_file.exists():
        return {}
    for line in score_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        numeric = {k: v for k, v in data.items() if isinstance(v, (int, float))}
        return numeric
    return {}


def extract_failures(result_file: Path, mode: str, analysis_dir: Path) -> Path | None:
    if not result_file.exists():
        return None

    failures: List[Dict[str, object]] = []
    with result_file.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            checker = entry.get("checker", {})
            is_correct = checker.get("is_correct")
            if is_correct is None:
                is_correct = entry.get("is_correct")
            if is_correct is True:
                continue
            reason = None
            if isinstance(checker, dict):
                reason = checker.get("reason")
            if not reason:
                reason = entry.get("result") or entry.get("error")
            failures.append(
                {
                    "id": entry.get("id"),
                    "status": "error" if "traceback" in entry else "incorrect",
                    "reason": reason,
                }
            )

    if not failures:
        return None

    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / f"{mode.lower()}_failures.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for row in failures:
            fh.write(json.dumps(row) + "\n")
    return out_path


def write_comparison(analysis_dir: Path, runs: List[RunSummary]) -> None:
    if len(runs) < 2:
        return

    status_map: Dict[str, Dict[str, str]] = defaultdict(dict)
    for summary in runs:
        failures: Dict[str, str] = {}
        if summary.failures_file and summary.failures_file.exists():
            for line in summary.failures_file.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                failures[record["id"]] = record["status"]
        # entries not listed are successes
        key = summary.mode.lower()
        for entry_id in failures:
            status_map[entry_id][key] = failures[entry_id]

    if not status_map:
        return

    comparison_path = analysis_dir / "comparison.json"
    with comparison_path.open("w", encoding="utf-8") as fh:
        json.dump(status_map, fh, indent=2)


def append_history(dest: Path, summary: RunSummary) -> None:
    history_path = dest / "run_history.jsonl"
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(summary.to_json()) + "\n")


def run_single_variant(
    dest: Path,
    env: Dict[str, str],
    cwd: Path,
    args: argparse.Namespace,
    label: str,
    model_id: str,
    subset: Path | None,
) -> RunSummary:
    logs_dir = dest / "logs"
    result_file, score_file = bfcl_paths(dest, model_id, args.category)

    generate_cmd = [
        sys.executable,
        "-m",
        "bfcl_eval._llm_response_generation",
        "--model",
        model_id,
        "--test-category",
        args.category,
        "--num-threads",
        str(args.threads),
        "--temperature",
        str(args.temperature),
    ]
    if subset is not None:
        generate_cmd.append("--run-ids")

    run_command(generate_cmd, env, logs_dir / f"generate_{label}.log", cwd)

    evaluate_cmd = [
        sys.executable,
        "-m",
        "bfcl_eval.eval_checker.eval_runner",
        "--model",
        model_id,
        "--test-category",
        args.category,
    ]
    if subset is not None:
        evaluate_cmd.append("--partial-eval")

    run_command(evaluate_cmd, env, logs_dir / f"evaluate_{label}.log", cwd)

    analysis_dir = dest / "analysis"
    failures_file = extract_failures(result_file, label, analysis_dir)
    score = load_score(score_file)

    return RunSummary(
        mode=label,
        model_id=model_id,
        category=args.category,
        mini=args.mini,
        score=score,
        result_file=result_file,
        score_file=score_file,
        failures_file=failures_file,
    )


def main() -> None:
    args = parse_args()
    dest = Path(args.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    env = ensure_openrouter_env(args)
    env["BFCL_PROJECT_ROOT"] = str(dest)

    subset_path = write_subset_file(dest, args.category, MINI_TEST_IDS) if args.mini else None

    runs: List[RunSummary] = []
    cwd = Path(__file__).resolve().parent / "berkeley-function-call-leaderboard"

    for label, model_id in resolve_model_ids(args.model, args.mode):
        summary = run_single_variant(dest, env, cwd, args, label, model_id, subset_path)
        append_history(dest, summary)
        runs.append(summary)

    write_comparison(dest / "analysis", runs)


if __name__ == "__main__":
    main()
