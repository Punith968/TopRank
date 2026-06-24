def generate_reasoning(row) -> str:
    parts = []
    semantic_sim = row.get('semantic_similarity', 0.5)
    yoe = row.get('years_of_experience', 0.0)
    is_only_consulting = row.get('is_only_consulting', 0.0)
    consulting_ratio = row.get('consulting_ratio', 0.0)
    job_hopper = row.get('job_hopper', 0.0)
    no_prod = row.get('no_production', 0.0)
    is_pure_research = row.get('is_pure_research', 0.0)
    keyword_stuffer = row.get('keyword_stuffer', 0.0)
    
    if semantic_sim > 0.65:
        parts.append(f"Strong semantic profile similarity ({semantic_sim:.2f}) to AI Engineering needs.")
    elif semantic_sim > 0.5:
        parts.append("Decent alignment with the required AI/ML domain.")
    else:
        parts.append("Relevant base skills with adjacent experience.")
        
    if yoe >= 5.0 and yoe <= 9.0:
        parts.append(f"Ideal range at {yoe:.1f} YoE.")
    else:
        parts.append(f"Has {yoe:.1f} YoE.")
        
    if is_only_consulting > 0.5:
        parts.append("Note: Career is entirely at IT consulting firms.")
    elif consulting_ratio < 0.2:
        parts.append("Strong product company career background.")
        
    if job_hopper > 0.5:
        parts.append("Has frequent job transition history.")
    if no_prod > 0.5:
        parts.append("Caution: Title timeline indicates potential coding gap recently.")
    if is_pure_research > 0.5:
        parts.append("Note: Career is primarily academic/research-focused.")
    if keyword_stuffer > 0.5:
        parts.append("Warning: High density of keywords in skills without matching career evidence.")
        
    return " ".join(parts)
