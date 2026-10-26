"""Run the prepared HKUST(GZ) Qwen continuation with a process-only API key.

The launcher tests one small provider response before entering the expensive
pipeline. It records no key and leaves the previous Qwen3.5 run untouched.
"""
from __future__ import annotations

import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import subprocess
import sys

from deepeye import runner
from deepeye.resume import DEFAULT_BASE_URL, DEFAULT_MODEL


def probe_school_api(key: str) -> dict:
    import httpx
    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=DEFAULT_BASE_URL, timeout=60, max_retries=0,
                    http_client=httpx.Client(trust_env=False))
    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
        n=1,
        temperature=0,
        max_tokens=128,
        extra_body={"enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}},
    )
    if len(response.choices) != 1 or not (response.choices[0].message.content or "").strip():
        raise RuntimeError("Provider returned no visible content for the one-choice probe")
    usage = response.usage
    return {"model": response.model, "choices": len(response.choices),
            "content_present": True, "finish_reason": response.choices[0].finish_reason,
            "response_preview": response.choices[0].message.content[:120],
            "prompt_tokens": usage.prompt_tokens if usage else None,
            "completion_tokens": usage.completion_tokens if usage else None}


def launch(run_name: str, *, execute: bool) -> None:
    key = os.environ.get("AIGC_API_KEY") or getpass("HKUST(GZ) AIGC API key: ")
    if not key:
        raise ValueError("AIGC_API_KEY is required; no network call was made")
    os.environ["AIGC_API_KEY"] = key
    os.environ["DEEPEYE_API_KEY"] = key
    env = os.environ.copy()
    env.update({"DEEPEYE_MODEL": DEFAULT_MODEL, "DEEPEYE_BASE_URL": DEFAULT_BASE_URL,
                "DEEPEYE_API_KEY": key, "AIGC_API_KEY": key})
    path = runner.run_dir(run_name)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest["model"] != DEFAULT_MODEL or manifest["base_url"] != DEFAULT_BASE_URL:
        raise ValueError("Prepared run does not match the official school model and V2 endpoint")
    result = probe_school_api(key)
    print("Provider probe passed: " + json.dumps(result, ensure_ascii=False), flush=True)
    if not execute:
        return
    for command in (
        [sys.executable, "-u", "-m", "deepeye.runner", "run", "--run-name", run_name],
        [sys.executable, "-u", "-m", "deepeye.runner", "evaluate", "--run-name", run_name],
        [str(runner.ROOT / ".venv" / "Scripts" / "python.exe"), "-u", "-m", "deepeye.analyze", "--run-name", run_name],
    ):
        subprocess.run(command, cwd=runner.ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="bird-train-300-hkust-qwen36-resume")
    parser.add_argument("--execute", action="store_true", help="After the probe, run pipeline, evaluation and error analysis")
    args = parser.parse_args()
    try:
        launch(args.run_name, execute=args.execute)
    except Exception as exc:
        print(runner.redact(f"{type(exc).__name__}: {exc}"), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
