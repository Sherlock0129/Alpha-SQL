"""Store a key locally without echoing it or putting it in shell history."""
from getpass import getpass
from pathlib import Path
import os
from dotenv import set_key


if __name__ == '__main__':
    target = Path(__file__).resolve().parents[1] / '.env.local'
    value = getpass('DASHSCOPE_API_KEY (hidden): ').strip()
    if not value or any(c.isspace() for c in value):
        raise SystemExit('Key must be non-empty and contain no whitespace')
    if not target.exists():
        descriptor = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(descriptor)
    target.chmod(0o600)
    set_key(str(target), 'DASHSCOPE_API_KEY', value)
    target.chmod(0o600)
    print('Project-local API key configured. Key value is not displayed.')
