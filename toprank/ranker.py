import numpy as np
import pandas as pd

WEIGHTS = {
    'semantic_similarity': 0.25,
    'career_trajectory': 0.25,
    'skill_relevance': 0.20,
    'behavioral_signals': 0.20,
    'availability': 0.10,
}

def compute_final_score(features_df: pd.DataFrame, similarities: np.ndarray) -> np.ndarray:
    scores = []
    for i, row in features_df.iterrows():
        sim = similarities[i]
        s_score = sim
        
        yoe = row['years_of_experience']
        yoe_suitability = np.exp(-0.5 * ((yoe - 7.0) / 2.0)**2)
        
        c_score = yoe_suitability
        if row['is_only_consulting'] > 0.5:
            c_score -= 0.3
        if row['job_hopper'] > 0.5:
            c_score -= 0.2
        if row['no_production'] > 0.5:
            c_score -= 0.4
        if row['is_pure_research'] > 0.5:
            c_score -= 0.2
        c_score = max(0.0, c_score)
        
        sk_count = row['skills_count']
        s_rel = min(1.0, np.log1p(sk_count) / np.log1p(30))
        if row['keyword_stuffer'] > 0.5:
            s_rel -= 0.5
        s_rel = max(0.0, s_rel)
        
        b_score = (
            row['recruiter_response_rate'] * 0.4 +
            row['interview_completion_rate'] * 0.3 +
            row['offer_acceptance_rate'] * 0.3
        )
        if row['has_github'] > 0.5:
            b_score += 0.1
        b_score = min(1.0, max(0.0, b_score))
        
        np_days = row['notice_period_days']
        if np_days <= 15:
            a_score = 1.0
        elif np_days <= 30:
            a_score = 0.8
        elif np_days <= 60:
            a_score = 0.5
        else:
            a_score = 0.2
        if row['open_to_work_flag'] > 0.5:
            a_score += 0.1
        a_score = min(1.0, max(0.0, a_score))
        
        fit_score = (
            s_score * WEIGHTS['semantic_similarity'] +
            c_score * WEIGHTS['career_trajectory'] +
            s_rel * WEIGHTS['skill_relevance'] +
            b_score * WEIGHTS['behavioral_signals'] +
            a_score * WEIGHTS['availability']
        )
        scores.append(fit_score)
        
    return np.array(scores)
