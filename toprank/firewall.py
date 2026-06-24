import pandas as pd
from datetime import datetime

def is_bachelor(degree_str):
    d = str(degree_str).strip().lower().replace('.', '')
    return d in {'be', 'bsc', 'btech', 'bs', 'bca', 'bba'} or 'bachelor' in d

def is_higher(degree_str):
    d = str(degree_str).strip().lower().replace('.', '')
    return d in {'me', 'msc', 'mtech', 'ms', 'phd', 'mba', 'mca', 'pgdm'} or 'master' in d

def detect_honeypot_record(row) -> bool:
    profile = row.get('profile', {}) or {}
    skills = row.get('skills', []) or []
    career = row.get('career_history', []) or []
    education = row.get('education', []) or []
    signals = row.get('redrob_signals', {}) or {}

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

    # Trap 3: Title inconsistency
    current_title = str(profile.get('current_title', '')).lower()
    if any(kw in current_title for kw in ['senior', 'lead', 'principal']) and yoe < 3:
        return True

    # Trap 4: Salary mismatch
    salary = signals.get('expected_salary_range_inr_lpa', {}) or {}
    if isinstance(salary, dict) and yoe < 5:
        max_sal = salary.get('max', 0) or 0
        if max_sal > 50:
            return True

    # Trap 5: PhD with YOE < 4
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

    # Trap 11: Career Pre-Graduation Mismatch
    bachelor_grad_year = None
    for edu in education:
        if isinstance(edu, dict):
            degree = edu.get('degree', '')
            if is_bachelor(degree):
                end_year = edu.get('end_year')
                if end_year:
                    bachelor_grad_year = end_year
                    break
    if bachelor_grad_year and career:
        start_years = []
        for job in career:
            if isinstance(job, dict) and job.get('start_date'):
                try:
                    syear = int(job['start_date'].split('-')[0])
                    start_years.append(syear)
                except (ValueError, IndexError):
                    pass
        if start_years:
            earliest_start_year = min(start_years)
            if earliest_start_year < (bachelor_grad_year - 4):
                return True

    # Trap 12: Degree Level Timeline Conflict
    bachelor_end = None
    higher_end = None
    for edu in education:
        if isinstance(edu, dict):
            degree = edu.get('degree', '')
            end_year = edu.get('end_year')
            if end_year:
                if is_bachelor(degree):
                    bachelor_end = end_year
                elif is_higher(degree):
                    higher_end = end_year
    if bachelor_end and higher_end and higher_end < bachelor_end:
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

    # Trap 14: Salary Min-Max Anomaly
    sal = signals.get('expected_salary_range_inr_lpa', {}) or {}
    if isinstance(sal, dict):
        s_min = sal.get('min')
        s_max = sal.get('max')
        if s_min is not None and s_max is not None and s_max < s_min:
            return True

    # Trap 15: Last Active before Signup date
    signup_str = signals.get('signup_date')
    active_str = signals.get('last_active_date')
    if signup_str and active_str:
        try:
            signup = datetime.strptime(signup_str, "%Y-%m-%d")
            active = datetime.strptime(active_str, "%Y-%m-%d")
            if active < signup:
                return True
        except Exception:
            pass

    # Trap 16: Company Founding Year Violation
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

    # Trap 17: Education Timeline Overlap Violation
    for i in range(len(education)):
        for j in range(i + 1, len(education)):
            edu1 = education[i]
            edu2 = education[j]
            if isinstance(edu1, dict) and isinstance(edu2, dict):
                inst1 = str(edu1.get('institution', '')).lower()
                inst2 = str(edu2.get('institution', '')).lower()
                if inst1 == inst2 or not inst1 or not inst2:
                    continue
                s1 = edu1.get('start_year')
                e1 = edu1.get('end_year')
                s2 = edu2.get('start_year')
                e2 = edu2.get('end_year')
                if s1 and e1 and s2 and e2:
                    if s1 <= e2 and s2 <= e1:
                        return True

    return False

def detect_honeypots(df: pd.DataFrame) -> pd.Series:
    return df.apply(detect_honeypot_record, axis=1)
