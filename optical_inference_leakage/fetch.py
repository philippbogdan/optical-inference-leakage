"""Download public inputs at pinned revisions without redistributing them."""
import argparse
import hashlib
from . import paths


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(split):
    """Fetch validation or test parquet and verify the recorded test checksum."""
    from huggingface_hub import hf_hub_download
    if split not in ('validation', 'test'):
        raise ValueError(f'Unsupported split: {split}')
    repository, revision = paths.DATASET
    path = hf_hub_download(repo_id=repository, revision=revision, repo_type='dataset',
                           filename=f'wikitext-2-raw-v1/{split}-00000-of-00001.parquet')
    if split == 'test' and sha256(path) != paths.TEST_SHA256:
        raise ValueError('WikiText-2 test checksum mismatch')
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=paths.LLM_REVS, help='Download one model; omit for all models')
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    for name in ([args.model] if args.model else paths.LLM_REVS):
        repository, revision = paths.LLM_REVS[name]
        path = snapshot_download(repo_id=repository, revision=revision,
                                 allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model', 'LICENSE*', 'README.md'])
        print(name, revision, path)
    for split in ('validation', 'test'):
        path = fetch(split)
        print(split, sha256(path))


if __name__ == '__main__':
    main()
