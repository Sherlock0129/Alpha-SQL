"""Bounded, non-streaming OpenAI-compatible calls for structured SQL output."""
import json
import os
import time
from pathlib import Path
from openai import OpenAI
from alphasql.llm_call.runtime import configure_environment
from alphasql.llm_call.cost_recoder import CostRecorder

DEFAULT_COST_RECORDER = CostRecorder(model='qwen3-coder-flash')
N_CALLING_STRATEGY_SINGLE = 'single'
N_CALLING_STRATEGY_MULTIPLE = 'multiple'
_requests = 0


def call_openai(prompt, model, temperature=0.0, top_p=1.0, n=1,
                max_tokens=512, stop=None, base_url=None, api_key=None,
                n_strategy='multiple', cost_recorder=DEFAULT_COST_RECORDER):
    global _requests
    configure_environment()
    if n < 1:
        raise ValueError('n must be positive')
    if n_strategy not in ('single', 'multiple'):
        raise ValueError('Unsupported sampling strategy')
    outputs = []
    with OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'),
                base_url=base_url or os.getenv('OPENAI_BASE_URL'),
                timeout=float(os.getenv('LLM_REQUEST_TIMEOUT', '60')),
                max_retries=int(os.getenv('LLM_MAX_RETRIES', '1'))) as client:
        for _ in range(n):
            if _requests >= int(os.getenv('LLM_MAX_REQUESTS', '120')):
                raise RuntimeError('LLM request budget exhausted in this process')
            _requests += 1
            started = time.monotonic()
            response = client.chat.completions.create(
                model=model, messages=[{'role': 'user', 'content': prompt}],
                temperature=temperature, top_p=top_p, n=1,
                max_tokens=max_tokens, stop=stop, stream=False)
            usage = response.usage
            if usage and cost_recorder is not None:
                cost_recorder.update_cost(usage.prompt_tokens, usage.completion_tokens)
            log_path = os.getenv('ALPHASQL_USAGE_PATH')
            if log_path:
                path = Path(log_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('a') as handle:
                    handle.write(json.dumps({'model': model, 'pid': os.getpid(),
                        'seconds': round(time.monotonic() - started, 3),
                        'prompt_tokens': usage.prompt_tokens if usage else None,
                        'completion_tokens': usage.completion_tokens if usage else None,
                        'finish_reason': response.choices[0].finish_reason}) + '\n')
            if response.choices[0].finish_reason == 'length':
                raise RuntimeError('Model output truncated; increase max_tokens')
            content = response.choices[0].message.content
            if not content:
                raise RuntimeError('Model returned empty content')
            outputs.append(content)
    return outputs
