# TopRank

[![Quality](https://github.com/Punith968/TopRank/actions/workflows/quality.yml/badge.svg)](https://github.com/Punith968/TopRank/actions/workflows/quality.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)

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

Built for the Redrob candidate-ranking hackathon: rank 100K synthetic candidate profiles against a *Senior AI Engineer* job description and return the top 100 with an explanation for each.

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
toprank/
  dataloader.py          JSONL / JSON / .gz candidate loading
  firewall.py            Anomaly ("honeypot") detection for impossible profiles
  features.py            Career, skill, behavioural and availability features
  ranker.py              Weighted composite fit score
  nlg.py                 Fact-based, per-candidate explanations
rank.py                  CLI pipeline (optional BGE cross-encoder rerank of the top 200)
precompute.py            Offline sentence-embedding similarities -> candidate_embeddings.npz
app.py                   Gradio interface
validate_submission.py   Output-format validator
data/                    Candidate schema and a 50-profile sample
docs/                    Solution architecture and diagram
submission/              Hackathon submission (ranking CSV, deck, metadata)
tests/                   pytest suite
```

`candidate_embeddings.npz` is stored with Git LFS. Run `git lfs pull` after cloning. Without it the pipeline still runs, using a neutral similarity of 0.5.

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

Try it on the bundled 50-profile sample (no model download needed):

```bash
python rank.py --candidates data/sample_candidates.json --out ranking.csv --no-rerank
```

On the full hackathon dataset, with cross-encoder reranking:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

`rank.py` accepts `.jsonl`, `.jsonl.gz` and `.json` (array) inputs. `--embeddings PATH` points to a different similarity file.

Run the interactive dashboard:

```bash
python app.py
```

Then open the local URL printed by Gradio.

## Scoring model

Each candidate that passes the firewall gets a weighted score (`toprank/ranker.py` is the source of truth):

| Component | Weight | Signals |
|---|---|---|
| Career trajectory | 0.45 | Gaussian fit around 7 years of experience; penalties for consulting-only careers, job hopping (3+ stints under 18 months), no current production role, pure research, LangChain-only or CV/speech-only skills, and non-technical titles |
| Semantic similarity | 0.25 | Profile-to-JD embedding similarity (precomputed), replaced by scaled BGE-reranker scores for the top 200 |
| Skill relevance | 0.15 | Log-scaled skill count, with a penalty for keyword stuffing (more than 40 skills) |
| Behavioural signals | 0.10 | Recruiter response, interview completion, offer acceptance, activity recency, GitHub presence |
| Availability | 0.05 | Notice period (≤15 / ≤30 / ≤60 / >60 days), open-to-work flag |

Ties are broken by `candidate_id`, and the top-100 scores are min-max normalized to [0, 1].

The firewall removes profiles with logically impossible data before scoring. Examples are expert skills used for 0 months, experience with a tool longer than the tool has existed, jobs at companies before they were founded, education that ends before it starts, and senior titles with under 3 years of experience.

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

## Testing

```bash
pip install pandas numpy pytest
python -m pytest -q tests
```

36 tests cover feature extraction (including null and zero-valued signals), every firewall trap, scoring properties (determinism, monotonicity in similarity and notice period, the experience curve), the end-to-end CLI on the sample data, the Git LFS pointer fallback, and the submission validator. GitHub Actions runs them on every push and pull request.

## Development

```bash
python validate_submission.py <submission.csv>
```

The committed `submission/Punith C.csv` was produced before the feature fixes in this revision (notice period 0 vs. missing, whole-word title matching). Re-running on the full dataset can therefore change a few positions.

Keep changes focused and document changes to scoring logic or validation rules.

## Fairness and limitations

Candidate ranking can affect real people. The current heuristic signals are project-specific and may encode undesirable proxies or assumptions. In particular, signals involving company type, job tenure, employment status, or career trajectory should be validated for relevance and fairness before any real hiring use.

This project should be treated as an engineering/research prototype, not as an autonomous hiring decision-maker.

## License

[MIT](LICENSE) © Punith C
