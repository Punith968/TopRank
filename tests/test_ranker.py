import numpy as np
import pandas as pd
import pytest

from tests.conftest import make_candidate
from toprank.features import engineer_features
from toprank.ranker import WEIGHTS, compute_final_score


def score(*candidates, sims=None):
    df = engineer_features(pd.DataFrame(list(candidates)))
    sims = np.full(len(df), 0.6) if sims is None else np.asarray(sims, dtype=float)
    return compute_final_score(df, sims)


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_scores_are_deterministic():
    c = make_candidate()
    assert score(c, c).tolist() == score(c, c).tolist()


def test_higher_semantic_similarity_ranks_higher():
    a, b = make_candidate("CAND_0000001"), make_candidate("CAND_0000002")
    s = score(a, b, sims=[0.9, 0.3])
    assert s[0] > s[1]
    assert s[0] - s[1] == pytest.approx(0.6 * WEIGHTS["semantic_similarity"])


def test_shorter_notice_period_scores_higher():
    s = score(make_candidate(redrob_signals__notice_period_days=0),
              make_candidate(redrob_signals__notice_period_days=90))
    assert s[0] > s[1]


def test_experience_curve_peaks_near_seven_years():
    s = score(make_candidate(profile__years_of_experience=7),
              make_candidate(profile__years_of_experience=1),
              make_candidate(profile__years_of_experience=15))
    assert s[0] > s[1] and s[0] > s[2]


def test_non_technical_profile_is_heavily_penalised():
    s = score(make_candidate(), make_candidate(profile__current_title="HR Manager"))
    assert s[0] - s[1] > 0.3


def test_scores_stay_in_unit_interval(sample_path):
    from toprank.dataloader import load_candidates

    df = engineer_features(load_candidates(str(sample_path)))
    s = compute_final_score(df, np.full(len(df), 0.5))
    assert np.all((s >= 0) & (s <= 1))
