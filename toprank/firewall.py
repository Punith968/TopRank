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
    # Trap 3: Title inconsistency (Senior/Lead/Principal roles with < 3 years total YoE)
    current_title = str(profile.get('current_title', '')).lower()
    if any(kw in current_title for kw in ['senior', 'lead', 'principal']) and yoe < 3:
        return True

    # Trap 4: Salary mismatch
    signals = row.get('redrob_signals', {}) or {}
    salary = signals.get('expected_salary_range_inr_lpa', {}) or {}
    if isinstance(salary, dict) and yoe < 5:
        max_sal = salary.get('max', 0) or 0
        if max_sal > 50:
            return True

    # Trap 5: PhD with YOE < 4
    education = row.get('education', []) or []
    for edu in education:
        if isinstance(edu, dict):
            degree = str(edu.get('degree', '')).lower().replace('.', '')
            if 'phd' in degree and yoe < 4:
                return True

    # Trap 6: Notice period impossible
    if signals.get('notice_period_days', 0) < 0:
        return True

    # Trap 7: Date sequence impossibility
    for edu in education:
        if isinstance(edu, dict):
            start = edu.get('start_year')
            end = edu.get('end_year')
            if start and end and end < start:
                return True
    # Trap 8: Skill duration exceeds YoE
    for skill in skills:
        if isinstance(skill, dict):
            dur = skill.get('duration_months', 0) or 0
            if yoe > 0 and dur > (yoe * 12 + 24):
                return True

    # Trap 9: Chronological Tech Release Impossibility
    recent_techs = {
        'qlora': 37, 'llamaindex': 43, 'langchain': 44, 'prompt engineering': 48,
        'chatgpt': 43, 'peft': 40, 'gpt-4': 39, 'gpt4': 39, 'bge': 34, 'e5': 42
    }
    for skill in skills:
        if isinstance(skill, dict):
            sname = str(skill.get('name', '')).lower()
            dur = skill.get('duration_months', 0) or 0
            for tech, max_dur in recent_techs.items():
                if tech in sname and dur > max_dur:
                    return True

    # Trap 10: Career Span vs YoE Mismatch
    if career and yoe > 0:
        start_dates = []
        for job in career:
            if isinstance(job, dict) and job.get('start_date'):
                try:
                    s_parts = job['start_date'].split('-')
                    if len(s_parts) == 3:
                        syear = int(s_parts[0])
                        smonth = int(s_parts[1])
                        elapsed_months = (2026 - syear) * 12 + (6 - smonth)
                        start_dates.append(elapsed_months)
                except (ValueError, IndexError):
                    pass
        if start_dates:
            max_elapsed = max(start_dates)
            if (yoe * 12) > (max_elapsed + 12):
                return True
    return False

def detect_honeypots(df: pd.DataFrame) -> pd.Series:
    return df.apply(detect_honeypot_record, axis=1)
