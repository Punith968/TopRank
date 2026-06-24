import gradio as gr
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from toprank import (
    detect_honeypot_record,
    engineer_features,
    compute_final_score,
    generate_reasoning
)

# Load candidate data
def load_data():
    try:
        path = "data/sample_candidates.json"
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

# Initialize model (cached globally)
print("Loading SentenceTransformer model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

JD_TEXT = """
Senior AI Engineer role at a Series A AI startup. We need deep technical depth in modern ML systems—embeddings, retrieval, ranking, LLMs, fine-tuning, and a scrappy product-engineering attitude.
Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5) deployed to real users.
Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).
"""
jd_embedding = model.encode([JD_TEXT], convert_to_numpy=True)

def get_ranked_candidates(yoe_min, yoe_max, search_query):
    candidates = load_data()
    if not candidates:
        return pd.DataFrame()
        
    df = pd.DataFrame(candidates)
    
    # Run firewall
    df['is_honeypot'] = df.apply(detect_honeypot_record, axis=1)
    # Mark scores for honeypots
    valid_df = df[~df['is_honeypot']].reset_index(drop=True)
    
    if len(valid_df) == 0:
        return pd.DataFrame(columns=["Rank", "ID", "Score", "YoE", "Title", "Reasoning"])
        
    # Feature extraction
    features_df = engineer_features(valid_df)
    
    # Compute similarity on-the-fly for sample data
    texts = []
    for cand in valid_df.to_dict('records'):
        profile = cand.get('profile', {}) or {}
        summary = profile.get('summary', '')
        summary_words = summary.split()[:35]
        parts = [
            profile.get('headline', ''),
            ' '.join(summary_words),
            profile.get('current_title', ''),
        ]
        for job in cand.get('career_history', []):
            if isinstance(job, dict):
                parts.append(job.get('title', ''))
        texts.append(' '.join(filter(None, parts)).lower())
        
    embeddings = model.encode(texts, convert_to_numpy=True)
    similarities = cosine_similarity(embeddings, jd_embedding).flatten()
    
    # Compute scores
    scores = compute_final_score(features_df, similarities)
    features_df['score'] = scores
    features_df['semantic_similarity'] = similarities
    
    # Sort
    features_df = features_df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    
    # Rescale scores to [0, 1] for display
    s_min = features_df['score'].min()
    s_max = features_df['score'].max()
    if s_max > s_min:
        features_df['score'] = (features_df['score'] - s_min) / (s_max - s_min)
    else:
        features_df['score'] = 1.0
    features_df['score'] = features_df['score'].round(4)
    
    # Sort again and assign ranks
    features_df = features_df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    features_df['rank'] = range(1, len(features_df) + 1)
    
    # Generate reasoning
    features_df['reasoning'] = features_df.apply(generate_reasoning, axis=1)
    
    # Apply UI filters
    rows = []
    for _, row in features_df.iterrows():
        yoe = row['years_of_experience']
        cid = row['candidate_id']
        
        cand_record = valid_df[valid_df['candidate_id'] == cid].iloc[0]
        title = (cand_record.get('profile', {}) or {}).get('current_title', 'Software Engineer')
        skills = ", ".join([s.get('name', '') for s in cand_record.get('skills', [])[:5] if isinstance(s, dict)])
        
        if not (yoe_min <= yoe <= yoe_max):
            continue
            
        if search_query:
            q = search_query.lower()
            if q not in title.lower() and q not in skills.lower():
                continue
                
        rows.append({
            "Rank": int(row['rank']),
            "Candidate ID": cid,
            "Fit Score": float(row['score']),
            "YoE": yoe,
            "Current Title": title,
            "Explanation / Justification": row['reasoning']
        })
        
    return pd.DataFrame(rows)

demo = gr.Interface(
    fn=get_ranked_candidates,
    inputs=[
        gr.Slider(0, 30, value=4, label="Min Years of Experience"),
        gr.Slider(0, 30, value=15, label="Max Years of Experience"),
        gr.Textbox(label="Filter by Title / Skills Keyword")
    ],
    outputs=gr.Dataframe(label="Screened & Ranked Candidates"),
    title="toprank Candidate Screening Web Dashboard",
    description="Interactive candidate ranking interface using a 17-stage Firewall and MiniLM semantic similarity."
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
