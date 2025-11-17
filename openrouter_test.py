#!/usr/bin/env python3
"""Minimal script to verify access to OpenRouter models."""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import requests


def run_request(model: str, prompt: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Please set OPENROUTER_API_KEY or OPENAI_API_KEY in the environment")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ],
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        data=json.dumps(payload),
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="openai/gpt-5-nano",
        help="Model ID to query via OpenRouter (default: %(default)s)",
    )
    parser.add_argument(
        "--prompt",
        default="Write a single haiku about open benchmarks.",
        help="Prompt text to send.",
    )
    args = parser.parse_args()

    try:
        data = run_request(args.model, args.prompt)
    except requests.HTTPError as exc:  # pragma: no cover - helpful debug output
        print(exc.response.text, file=sys.stderr)
        raise

    choice = data["choices"][0]
    message = choice["message"]
    content = message.get("content")
    if isinstance(content, list):
        output = "\n".join(block.get("text", "") for block in content if isinstance(block, dict))
    else:
        output = content or ""

    print("Model:", args.model)
    print("Prompt:", args.prompt)
    print("Response:\n" + output.strip())


if __name__ == "__main__":
    main()
