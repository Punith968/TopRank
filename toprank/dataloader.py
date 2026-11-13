import gzip
import json

import pandas as pd


def load_candidates(candidates_path: str) -> pd.DataFrame:
    """Load candidate profiles from JSON Lines (.jsonl / .jsonl.gz) or a JSON array (.json)."""
    open_func = gzip.open if candidates_path.endswith('.gz') else open
    with open_func(candidates_path, 'rt', encoding='utf-8') as f:
        if candidates_path.endswith(('.json', '.json.gz')):
            records = json.load(f)
        else:
            records = [json.loads(line) for line in (l.strip() for l in f) if line]
    return pd.DataFrame(records)
