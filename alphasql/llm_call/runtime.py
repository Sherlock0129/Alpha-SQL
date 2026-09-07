"""Project-scoped credentials. Shell environment takes precedence."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]


def configure_environment():
    load_dotenv(ROOT / '.env.local', override=False)
    key = os.getenv('DASHSCOPE_API_KEY')
    if key:
        os.environ['OPENAI_API_KEY'] = key
        os.environ['EMBEDDING_API_KEY'] = os.getenv('EMBEDDING_API_KEY') or key
        base = os.getenv('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
        os.environ['OPENAI_BASE_URL'] = base
        os.environ['EMBEDDING_BASE_URL'] = os.getenv('EMBEDDING_BASE_URL') or base


configure_environment()
