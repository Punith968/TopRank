import pandas as pd
from datetime import datetime

def is_bachelor(degree_str):
    d = str(degree_str).strip().lower().replace('.', '')
    return d in {'be', 'bsc', 'btech', 'bs', 'bca', 'bba'} or 'bachelor' in d

def is_higher(degree_str):
    d = str(degree_str).strip().lower().replace('.', '')
    return d in {'me', 'msc', 'mtech', 'ms', 'phd', 'mba', 'mca', 'pgdm'} or 'master' in d

def detect_honeypot_record(row) -> bool:
    """Conservative honeypot detection — flags only clearly impossible profiles.

    The spec states ~80 honeypots with "subtly impossible" conditions.
    We keep only traps that detect logically impossible data (fabricated
    timelines, contradictory durations), and AVOID traps that could flag
    legitimate edge cases in a 100K synthetic dataset (e.g. overlapping
    education, salary data noise, fresh PhD graduates).
    """
    profile = row.get('profile', {}) or {}
    skills = row.get('skills', []) or []
    career = row.get('career_history', []) or []
    education = row.get('education', []) or []
    signals = row.get('redrob_signals', {}) or {}

    # Trap 1: Expert skill with 0 duration_months — clearly impossible
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

    # Trap 3: Title inconsistency — Senior/Lead/Principal with <3 YoE
    current_title = str(profile.get('current_title', '')).lower()
    if any(kw in current_title for kw in ['senior', 'lead', 'principal']) and yoe < 3:
        return True

    # Trap 6: Notice period impossible (negative)
    if signals.get('notice_period_days', 0) < 0:
        return True

    # Trap 7: Date sequence impossibility — education end < start
    for edu in education:
        if isinstance(edu, dict):
            start = edu.get('start_year')
            end = edu.get('end_year')
            if start and end and end < start:
                return True

    # Trap 8: Skill duration far exceeds YoE
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

    # Trap 13: Expert/Advanced Skill Low Duration
    for skill in skills:
        if isinstance(skill, dict):
            prof_level = str(skill.get('proficiency', '')).lower()
            dur = skill.get('duration_months', 0) or 0
            if prof_level == 'expert' and dur < 6:
                return True
            elif prof_level == 'advanced' and dur < 3:
                return True

    # Trap 16: Company Founding Year Violation — clearly fabricated
    company_founding_years = {
        'pinecone': 2019, 'qdrant': 2021, 'weaviate': 2019, 'openai': 2015,
        'langchain': 2022, 'llamaindex': 2022, 'cohere': 2019, 'anthropic': 2021,
        'hugging face': 2016, 'huggingface': 2016, 'sarvam ai': 2023, 'sarvam.ai': 2023,
        'krutrim': 2023, 'mistral': 2023, 'perplexity': 2022, 'midjourney': 2021,
        'anyscale': 2019, 'together ai': 2022, 'together.ai': 2022
    }
    for job in career:
        if isinstance(job, dict):
            company = str(job.get('company', '')).lower()
            start_date = job.get('start_date')
            if start_date:
                try:
                    syear = int(start_date.split('-')[0])
                    for comp, fyear in company_founding_years.items():
                        if comp in company and syear < fyear:
                            return True
                except (ValueError, IndexError):
                    pass

    return False

def detect_honeypots(df: pd.DataFrame) -> pd.Series:
    return df.apply(detect_honeypot_record, axis=1)
