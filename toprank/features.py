import pandas as pd
from datetime import datetime

CONSULTING_FIRMS = {
    'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini',
    'tech mahindra', 'mindtree', 'hcl', 'mphasis', 'deloitte', 'ey',
    'pwc', 'kpmg', 'ibm'
}

RECENT_FRAMEWORK_SKILLS = {
    'langchain', 'llamaindex', 'openai api', 'chatgpt', 'gpt-4',
    'prompt engineering', 'autogen', 'crewai'
}

PRE_LLM_ML_SKILLS = {
    'machine learning', 'deep learning', 'nlp', 'neural network',
    'pytorch', 'tensorflow', 'scikit-learn', 'xgboost', 'recommendation',
    'search', 'ranking', 'retrieval', 'embeddings', 'bert', 'transformers',
    'information retrieval', 'natural language', 'text mining', 'data science'
}

CV_SPEECH_KEYWORDS = {
    'computer vision', 'image processing', 'object detection',
    'image segmentation', 'speech recognition', 'speech synthesis',
    'audio processing', 'robotics', 'autonomous', 'self-driving',
    'lidar', 'slam', 'pose estimation', 'image classification'
}

NLP_IR_KEYWORDS = {
    'nlp', 'search', 'retrieval', 'ranking', 'recommendation',
    'text', 'embeddings', 'information retrieval', 'natural language',
    'question answering', 'rag', 'vector'
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

    # NEW: LangChain-only check (JD explicitly penalizes this)
    skill_names_lower = [str(s.get('name', '')).lower() for s in skills if isinstance(s, dict)]
    has_recent_frameworks = any(any(rf in sn for rf in RECENT_FRAMEWORK_SKILLS) for sn in skill_names_lower)
    has_pre_llm_ml = any(any(pl in sn for pl in PRE_LLM_ML_SKILLS) for sn in skill_names_lower)
    is_langchain_only = 1.0 if (has_recent_frameworks and not has_pre_llm_ml) else 0.0

    # NEW: CV/Speech domain detection (JD explicitly deprioritizes)
    has_cv_speech = any(any(kw in sn for kw in CV_SPEECH_KEYWORDS) for sn in skill_names_lower)
    has_nlp_ir = any(any(kw in sn for kw in NLP_IR_KEYWORDS) for sn in skill_names_lower)
    is_cv_speech_only = 1.0 if (has_cv_speech and not has_nlp_ir) else 0.0

    # NEW: Engagement recency from last_active_date
    engagement_score = 0.5
    last_active = signals.get('last_active_date')
    if last_active:
        try:
            active_date = datetime.strptime(str(last_active), '%Y-%m-%d')
            months_since = (datetime(2026, 6, 1) - active_date).days / 30.0
            engagement_score = max(0.0, min(1.0, 1.0 - (months_since / 12.0)))
        except (ValueError, TypeError):
            engagement_score = 0.5

    # NEW: Non-technical role detection to prevent keyword-stuffing exploit
    current_title = str(profile.get('current_title', '')).lower()
    non_tech_kws = [
        'hr', 'human resources', 'recruiter', 'talent acquisition',
        'sales', 'marketing', 'accountant', 'accounting',
        'mechanical', 'civil', 'designer', 'writer', 'support',
        'operations', 'project manager', 'business analyst', 'finance', 'financial',
        'product manager', 'customer support', 'graphic designer'
    ]
    is_non_tech = 1.0 if any(kw in current_title for kw in non_tech_kws) else 0.0

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
        'is_langchain_only': is_langchain_only,
        'is_cv_speech_only': is_cv_speech_only,
        'is_non_tech': is_non_tech,
        'engagement_score': engagement_score,
        'skills_count': len(skills),
        'recruiter_response_rate': float(signals.get('recruiter_response_rate', 0.0) or 0.0),
        'interview_completion_rate': float(signals.get('interview_completion_rate', 0.0) or 0.0),
        'offer_acceptance_rate': float(signals.get('offer_acceptance_rate')) if signals.get('offer_acceptance_rate') is not None and float(signals.get('offer_acceptance_rate')) >= 0.0 else 0.475,
        'notice_period_days': float(signals.get('notice_period_days', 90.0) or 90.0),
        'open_to_work_flag': 1.0 if signals.get('open_to_work_flag') else 0.0
    }

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    features = [extract_features_record(row) for _, row in df.iterrows()]
    return pd.DataFrame(features)
