import csv
import subprocess
import sys
from pathlib import Path

from validate_submission import validate_submission

ROOT = Path(__file__).resolve().parents[1]


def run_rank(tmp_path, sample_path, *extra):
    out = tmp_path / "ranked.csv"
    result = subprocess.run(
        [sys.executable, str(ROOT / "rank.py"), "--candidates", str(sample_path),
         "--out", str(out), "--no-rerank", *extra],
        capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with open(out, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f)), result.stdout


def expected_rows(sample_path):
    from toprank.dataloader import load_candidates
    from toprank.firewall import detect_honeypots

    df = load_candidates(str(sample_path))
    return len(df) - int(detect_honeypots(df).sum())


def test_cli_ranks_sample_candidates(tmp_path, sample_path):
    rows, stdout = run_rank(tmp_path, sample_path)
    n = expected_rows(sample_path)
    assert 0 < n <= 50
    assert len(rows) == n  # honeypots are removed before ranking
    assert [int(r["rank"]) for r in rows] == list(range(1, n + 1))
    scores = [float(r["score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == 1.0
    assert all(r["reasoning"].strip() for r in rows)
    assert len({r["candidate_id"] for r in rows}) == n


def test_cli_output_is_deterministic(tmp_path, sample_path):
    a, _ = run_rank(tmp_path, sample_path)
    b, _ = run_rank(tmp_path, sample_path)
    assert a == b


def test_cli_survives_git_lfs_pointer(tmp_path, sample_path):
    pointer = tmp_path / "candidate_embeddings.npz"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    rows, stdout = run_rank(tmp_path, sample_path, "--embeddings", str(pointer))
    assert len(rows) == expected_rows(sample_path)
    assert "git lfs pull" in stdout


def test_validator_accepts_the_committed_submission():
    assert validate_submission(ROOT / "submission" / "Punith C.csv") == []


def test_validator_rejects_wrong_row_count(tmp_path, sample_path):
    rows, _ = run_rank(tmp_path, sample_path)  # only 50 rows
    out = tmp_path / "ranked.csv"
    errors = validate_submission(out)
    assert any("100" in e for e in errors)
