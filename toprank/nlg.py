JD_SKILL_KEYWORDS = {
    'embeddings', 'retrieval', 'ranking', 'llm', 'fine-tuning', 'finetuning',
    'sentence-transformers', 'bge', 'e5', 'pinecone', 'weaviate', 'qdrant',
    'milvus', 'opensearch', 'elasticsearch', 'faiss', 'vector', 'rag',
    'ndcg', 'mrr', 'nlp', 'transformers', 'pytorch', 'tensorflow',
    'python', 'machine learning', 'deep learning', 'bert', 'search',
    'recommendation', 'huggingface', 'information retrieval', 'a/b test',
    'xgboost', 'learning to rank'
}


def generate_reasoning(row) -> str:
    """Generate a fact-based, candidate-specific reasoning string.
    
    Expects the row to contain both engineered features AND raw candidate
    data columns (profile, skills, career_history, redrob_signals) that
    were merged back from the original DataFrame.
    """
    parts = []

    # --- Raw candidate data ---
    profile = row.get('profile', {}) or {}
    career = row.get('career_history', []) or []
    skills_raw = row.get('skills', []) or []
    signals = row.get('redrob_signals', {}) or {}

    current_title = str(profile.get('current_title', '') or '').strip()
    yoe = float(profile.get('years_of_experience', 0) or 0)
    semantic_sim = row.get('semantic_similarity', 0.5)

    # --- Engineered features ---
    is_only_consulting = row.get('is_only_consulting', 0)
    consulting_ratio = row.get('consulting_ratio', 0)
    job_hopper = row.get('job_hopper', 0)
    no_production = row.get('no_production', 0)
    is_pure_research = row.get('is_pure_research', 0)
    keyword_stuffer = row.get('keyword_stuffer', 0)
    is_langchain_only = row.get('is_langchain_only', 0)
    is_cv_speech_only = row.get('is_cv_speech_only', 0)

    # --- Derived helpers ---
    current_company = ''
    for job in (career if isinstance(career, list) else []):
        if isinstance(job, dict) and job.get('is_current'):
            current_company = str(job.get('company', '')).strip()
            break

    skill_names = [str(s.get('name', '')) for s in skills_raw if isinstance(s, dict)]
    matching = [s for s in skill_names
                if any(kw in s.lower() for kw in JD_SKILL_KEYWORDS)]

    # ================================================================
    # 1. Current role (always first — specific fact)
    # ================================================================
    if current_title and current_company:
        parts.append(f'Currently {current_title} at {current_company}.')
    elif current_title:
        parts.append(f'Working as {current_title}.')

    # ================================================================
    # 2. Semantic alignment (quantified)
    # ================================================================
    if semantic_sim > 0.65:
        parts.append(f'High profile-to-JD semantic match ({semantic_sim:.2f}) indicating strong retrieval/ranking alignment.')
    elif semantic_sim > 0.50:
        parts.append(f'Moderate semantic alignment ({semantic_sim:.2f}) with the Senior AI Engineer role.')
    else:
        parts.append(f'Lower semantic overlap ({semantic_sim:.2f}) with core JD requirements.')

    # ================================================================
    # 3. Years-of-experience assessment (JD: 5-9)
    # ================================================================
    if 5.0 <= yoe <= 9.0:
        parts.append(f'{yoe:.0f} YoE sits in the JD\'s ideal 5\u20139 year band.')
    elif yoe < 5.0:
        parts.append(f'At {yoe:.0f} YoE, slightly junior for the JD\'s 5\u20139 year target.')
    else:
        parts.append(f'At {yoe:.0f} YoE, experienced but the JD emphasises hands-on coding over seniority.')

    # ================================================================
    # 4. JD-matching skills (specific, named)
    # ================================================================
    if len(matching) >= 5:
        parts.append(f'Strong JD skill coverage including {matching[0]}, {matching[1]}, {matching[2]}, and {matching[3]}.')
    elif len(matching) >= 2:
        parts.append(f'Relevant skills: {", ".join(matching[:3])}.')
    elif len(matching) == 1:
        parts.append(f'Has {matching[0]} experience but limited additional JD-matching skills.')
    else:
        parts.append('Few directly JD-matching skills listed.')

    # ================================================================
    # 5. Career trajectory (product vs consulting)
    # ================================================================
    if is_only_consulting > 0.5:
        parts.append('Concern: career entirely at IT services/consulting firms; JD explicitly prefers product-company backgrounds.')
    elif consulting_ratio > 0.6:
        parts.append('Majority consulting-firm tenure; JD leans towards product-company experience.')
    elif consulting_ratio < 0.2 and is_only_consulting < 0.5:
        parts.append('Positive: primarily product-company career trajectory.')

    # ================================================================
    # 6. Honest concerns (JD-linked)
    # ================================================================
    if job_hopper > 0.5:
        parts.append('Concern: average tenure under 18 months; JD seeks 3+ year commitment.')
    if is_pure_research > 0.5:
        parts.append('Risk: purely research/academic career; JD requires production deployment track record.')
    if is_langchain_only > 0.5:
        parts.append('Flag: AI experience appears limited to recent LLM frameworks without deeper pre-LLM ML work.')
    if is_cv_speech_only > 0.5:
        parts.append('Note: primary expertise in CV/speech/robotics; JD focuses on NLP and information retrieval.')
    if keyword_stuffer > 0.5:
        parts.append('Caution: unusually high skill count (40+) without proportional career evidence.')
    if no_production > 0.5:
        parts.append('Gap: no current role listed; JD requires active coding in last 18 months.')

    # ================================================================
    # 7. Availability (notice period)
    # ================================================================
    notice = float(signals.get('notice_period_days', 90) or 90)
    if notice <= 30:
        parts.append(f'Available in {int(notice)} days, matching JD\'s sub-30-day preference.')
    elif notice > 60:
        parts.append(f'Notice period of {int(notice)} days may delay onboarding.')

    return ' '.join(parts)
