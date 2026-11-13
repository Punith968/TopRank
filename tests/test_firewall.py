import pytest

from tests.conftest import make_candidate
from toprank.firewall import detect_honeypot_record, is_bachelor, is_higher


def test_clean_profile_passes(candidate):
    assert detect_honeypot_record(candidate) is False


@pytest.mark.parametrize("overrides", [
    # expert skill with zero months of use
    {"skills": [{"name": "Python", "proficiency": "expert", "duration_months": 0}]},
    # senior title with < 3 years of experience
    {"profile__current_title": "Senior ML Engineer", "profile__years_of_experience": 2},
    # negative notice period
    {"redrob_signals__notice_period_days": -10},
    # education ends before it starts
    {"education": [{"degree": "B.Tech", "start_year": 2018, "end_year": 2015}]},
    # LangChain experience longer than LangChain has existed
    {"skills": [{"name": "LangChain", "proficiency": "advanced", "duration_months": 80}]},
    # job at a company before it was founded
    {"career_history": [{"company": "Anthropic", "title": "Engineer", "start_date": "2018-01-01",
                         "duration_months": 24, "is_current": True}]},
    # career history much longer than stated experience
    {"profile__years_of_experience": 2, "career_history": [
        {"company": "A", "title": "Engineer", "start_date": "2016-01-01", "duration_months": 120,
         "is_current": True}]},
])
def test_impossible_profiles_are_flagged(overrides):
    assert detect_honeypot_record(make_candidate(**overrides)) is True


def test_missing_and_null_fields_do_not_crash():
    c = make_candidate(redrob_signals__notice_period_days=None)
    c["skills"].append({"name": "SQL", "proficiency": "beginner", "duration_months": None})
    assert detect_honeypot_record(c) is False
    assert detect_honeypot_record({"candidate_id": "CAND_0000009"}) is False


def test_degree_helpers():
    assert is_bachelor("B.Tech") and is_bachelor("Bachelor of Science")
    assert is_higher("M.Tech") and is_higher("PhD") and not is_higher("B.E.")
