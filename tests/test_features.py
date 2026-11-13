import pytest

from tests.conftest import make_candidate
from toprank.features import engineer_features, extract_features_record


def test_clean_candidate_has_no_penalty_flags(candidate):
    f = extract_features_record(candidate)
    for flag in ["is_only_consulting", "job_hopper", "no_production", "is_pure_research",
                 "keyword_stuffer", "is_langchain_only", "is_cv_speech_only", "is_non_tech"]:
        assert f[flag] == 0.0, flag
    assert f["has_github"] == 1.0
    assert f["years_of_experience"] == 7.0


def test_zero_notice_period_means_immediate_joiner():
    # Regression: `value or 90` turned 0 days into 90 days.
    f = extract_features_record(make_candidate(redrob_signals__notice_period_days=0))
    assert f["notice_period_days"] == 0.0


def test_missing_notice_period_defaults_to_90():
    c = make_candidate()
    del c["redrob_signals"]["notice_period_days"]
    assert extract_features_record(c)["notice_period_days"] == 90.0


def test_null_signals_do_not_crash():
    c = make_candidate(redrob_signals__github_activity_score=None,
                       redrob_signals__offer_acceptance_rate=None,
                       redrob_signals__recruiter_response_rate=None)
    f = extract_features_record(c)
    assert f["has_github"] == 0.0
    assert f["offer_acceptance_rate"] == 0.475  # population prior
    assert f["recruiter_response_rate"] == 0.0


def test_consulting_only_career_is_flagged():
    c = make_candidate(career_history=[
        {"company": "Infosys", "title": "Engineer", "duration_months": 30, "is_current": True},
        {"company": "TCS", "title": "Engineer", "duration_months": 30},
    ])
    f = extract_features_record(c)
    assert f["is_only_consulting"] == 1.0
    assert f["consulting_ratio"] == 1.0


def test_job_hopper_needs_three_short_stints():
    short = [{"company": f"Co{i}", "title": "Engineer", "duration_months": 10} for i in range(3)]
    short[0]["is_current"] = True
    assert extract_features_record(make_candidate(career_history=short))["job_hopper"] == 1.0
    assert extract_features_record(make_candidate(career_history=short[:2]))["job_hopper"] == 0.0


def test_langchain_only_skills_are_flagged():
    c = make_candidate(skills=[{"name": "LangChain"}, {"name": "Prompt Engineering"}])
    assert extract_features_record(c)["is_langchain_only"] == 1.0


@pytest.mark.parametrize("title,expected", [
    ("HR Business Partner", 1.0),
    ("Senior Recruiter", 1.0),
    ("Product Manager", 1.0),
    ("Threat Analyst", 0.0),          # substring "hr" must not match
    ("Supportability Engineer", 0.0), # substring "support" must not match
    ("Machine Learning Engineer", 0.0),
])
def test_non_technical_title_detection_uses_whole_words(title, expected):
    c = make_candidate(profile__current_title=title)
    assert extract_features_record(c)["is_non_tech"] == expected


def test_engineer_features_returns_one_row_per_candidate(sample_path):
    from toprank.dataloader import load_candidates

    df = load_candidates(str(sample_path))
    feats = engineer_features(df)
    assert len(feats) == len(df) == 50
    assert feats["candidate_id"].is_unique
