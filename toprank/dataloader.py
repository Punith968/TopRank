import json
import gzip
import pandas as pd

def load_candidates(candidates_path: str) -> pd.DataFrame:
    records = []
    open_func = gzip.open if candidates_path.endswith('.gz') else open
    with open_func(candidates_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return pd.DataFrame(records)
