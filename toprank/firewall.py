import pandas as pd
from datetime import datetime

def detect_honeypot_record(row) -> bool:
    profile = row.get('profile', {}) or {}
    skills = row.get('skills', []) or []
    career = row.get('career_history', []) or []

    # Trap 1: Expert skill with 0 duration_months
    for skill in skills:
        if isinstance(skill, dict):
            if skill.get('proficiency') == 'expert' and skill.get('duration_months', 0) == 0:
                return True

    # Trap 2: Experience vs career history duration inconsistency
    total_career_months = sum(job.get('duration_months', 0) for job in career if isinstance(job, dict))
    yoe = float(profile.get('years_of_experience', 0) or 0)
    if yoe > 0:
        if total_career_months > (yoe * 12 + 36):
            return True
    return False

def detect_honeypots(df: pd.DataFrame) -> pd.Series:
    return df.apply(detect_honeypot_record, axis=1)
