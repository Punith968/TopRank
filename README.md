# TopRank

A deterministic candidate screening and ranking engine built around a multi-stage retrieval and scoring pipeline.

> **Important:** Benchmark figures in this README are project measurements, not universal guarantees. Re-run the benchmark on your hardware and dataset before using them as performance claims.

## What it does

TopRank processes candidate profiles through:

1. Profile validation and anomaly checks
2. Feature extraction
3. Pre-computed semantic similarity
4. Composite fit scoring
5. Deterministic tie-breaking
6. Ranked output generation
7. Candidate-specific, fact-based explanations

The scoring pipeline is intentionally deterministic: the same inputs and configuration produce the same ordering.

## Architecture

```text
Candidate JSONL
     |
     v
Validation / anomaly checks
     |
     v
Feature extraction ----> Semantic similarity index
     |                           |
     +-------------+-------------+
                   |
                   v
             Composite scorer
                   |
                   v
            Deterministic sort
                   |
                   v
          Ranked CSV + explanations
```

## Repository structure

```text
data/                    Dataset schemas and small samples
docs/                    Design documentation
toprank/                 Core ranking engine
app.py                   Gradio interface
rank.py                  CLI pipeline
precompute.py            Offline embedding preparation
validate_submission.py   Output validation
requirements.txt         Python dependencies
```

Large/private candidate datasets are not required for the sample workflow.

## Quick start

```bash
git clone https://github.com/Punith968/TopRank.git
cd TopRank
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

Run the pipeline with an available candidate dataset:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

Run the interactive dashboard:

```bash
python app.py
```

Then open the local URL printed by Gradio.

## Scoring model

The repository's current scoring configuration combines semantic, career, skill, behavioural, and availability signals. The exact implementation in `toprank/ranker.py` is the source of truth; update this README when weights or features change.

## Evaluation

For reproducible evaluation, record:

- candidate dataset/version
- hardware and Python version
- preprocessing configuration
- whether semantic embeddings were precomputed
- wall-clock runtime
- ranking-quality metrics
- validation/firewall results

Avoid interpreting runtime or ranking quality from one machine or one dataset as a general guarantee.

## Development

Useful checks:

```bash
python -m compileall -q .
python validate_submission.py <submission.csv>
```

Keep changes focused and document changes to scoring logic or validation rules.

## Fairness and limitations

Candidate ranking can affect real people. The current heuristic signals are project-specific and may encode undesirable proxies or assumptions. In particular, signals involving company type, job tenure, employment status, or career trajectory should be validated for relevance and fairness before any real hiring use.

This project should be treated as an engineering/research prototype, not as an autonomous hiring decision-maker.

## License

See `LICENSE`.
