import pandas as pd

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'tech mahindra', 'mindtree', 'hcl', 'mphasis', 'deloitte', 'ey',
    'pwc', 'kpmg', 'ibm'
}

def extract_features_record(row) -> dict:
    profile = row.get('profile', {}) or {}
    career = row.get('career_history', []) or []
    skills = row.get('skills', []) or []
    signals = row.get('redrob_signals', {}) or {}

    yoe = float(profile.get('years_of_experience', 0) or 0)
    
    total_jobs = len(career)
    consulting_jobs = 0
    for job in career:
        if isinstance(job, dict):
            comp = str(job.get('company', '')).lower()
            if any(cf in comp for cf in CONSULTING_FIRMS):
                consulting_jobs += 1
    
    consulting_ratio = consulting_jobs / total_jobs if total_jobs > 0 else 0.0
    is_only_consulting = 1.0 if (total_jobs > 0 and consulting_jobs == total_jobs) else 0.0

    tenures = [job.get('duration_months', 0) or 0 for job in career if isinstance(job, dict)]
    avg_tenure = sum(tenures) / len(tenures) if tenures else 24.0
    job_hopper = 1.0 if (avg_tenure < 18.0 and len(tenures) >= 3) else 0.0

    has_github = 1.0 if signals.get('github_activity_score', -1) >= 0 else 0.0
    no_production = 0.0
    if total_jobs > 0:
        current_jobs = [job for job in career if isinstance(job, dict) and job.get('is_current')]
        if not current_jobs:
            no_production = 1.0

    is_pure_research = 0.0
    research_titles = sum(1 for job in career if isinstance(job, dict) and any(kw in str(job.get('title', '')).lower() for kw in ['researcher', 'scientist', 'postdoc', 'academic']))
    if total_jobs > 0 and research_titles == total_jobs:
        is_pure_research = 1.0

    keyword_stuffer = 0.0
    if len(skills) > 40:
        keyword_stuffer = 1.0

    return {
        'candidate_id': row.get('candidate_id'),
        'years_of_experience': yoe,
        'consulting_ratio': consulting_ratio,
        'is_only_consulting': is_only_consulting,
        'job_hopper': job_hopper,
        'has_github': has_github,
        'no_production': no_production,
        'is_pure_research': is_pure_research,
        'keyword_stuffer': keyword_stuffer,
        'skills_count': len(skills),
        'recruiter_response_rate': float(signals.get('recruiter_response_rate', 0.0) or 0.0),
        'interview_completion_rate': float(signals.get('interview_completion_rate', 0.0) or 0.0),
        'offer_acceptance_rate': float(signals.get('offer_acceptance_rate', 0.0) or 0.0),
        'notice_period_days': float(signals.get('notice_period_days', 90.0) or 90.0),
        'open_to_work_flag': 1.0 if signals.get('open_to_work_flag') else 0.0
    }

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [extract_features_record(row) for _, row in df.iterrows()]
    return pd.DataFrame(features)
