import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def make_candidate(cid="CAND_0000001", **overrides):
    """A clean, plausible profile that passes every firewall check."""
    cand = {
        "candidate_id": cid,
        "profile": {
            "headline": "ML Engineer | Search & Ranking",
            "summary": "Builds retrieval and ranking systems.",
            "years_of_experience": 7.0,
            "current_title": "Machine Learning Engineer",
            "current_company": "Razorpay",
        },
        "career_history": [
            {"company": "Razorpay", "title": "Machine Learning Engineer", "start_date": "2022-01-01",
             "duration_months": 48, "is_current": True},
            {"company": "Flipkart", "title": "Software Engineer", "start_date": "2019-06-01",
             "duration_months": 30, "is_current": False},
        ],
        "skills": [
            {"name": "Information Retrieval", "proficiency": "expert", "duration_months": 60},
            {"name": "Elasticsearch", "proficiency": "advanced", "duration_months": 48},
            {"name": "PyTorch", "proficiency": "advanced", "duration_months": 40},
        ],
        "education": [{"degree": "B.Tech", "start_year": 2013, "end_year": 2017}],
        "redrob_signals": {
            "recruiter_response_rate": 0.8,
            "interview_completion_rate": 0.9,
            "offer_acceptance_rate": 0.6,
            "notice_period_days": 30,
            "open_to_work_flag": True,
            "github_activity_score": 70,
            "last_active_date": "2026-05-01",
        },
    }
    for path, value in overrides.items():
        target = cand
        keys = path.split("__")
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value
    return copy.deepcopy(cand)


@pytest.fixture
def candidate():
    return make_candidate()


@pytest.fixture
def sample_path():
    return ROOT / "data" / "sample_candidates.json"
