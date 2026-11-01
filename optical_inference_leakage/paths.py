"""Immutable public input revisions and a local download cache."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('HF_HOME', str(ROOT / '.cache' / 'huggingface'))
LLM_REVS = {
    'Qwen3-0.6B': ('Qwen/Qwen3-0.6B-Base', 'da87bfb608c14b7cf20ba1ce41287e8de496c0cd'),
    'Qwen3-1.7B': ('Qwen/Qwen3-1.7B-Base', 'ea980cb0a6c2ae4b936e82123acc929f1cec04c1'),
    'Qwen3-4B': ('Qwen/Qwen3-4B-Base', '906bfd4b4dc7f14ee4320094d8b41684abff8539'),
    'Qwen3-8B': ('Qwen/Qwen3-8B-Base', '49e3418fbbbca6ecbdf9608b4d22e5a407081db4'),
}
DATASET = ('Salesforce/wikitext', 'b08601e04326c79dfdd32d625aee71d232d685c3')
TEST_SHA256 = '5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91'
