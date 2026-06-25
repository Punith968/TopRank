#!/usr/bin/env python3
"""
Deterministic Candidate Screening & Ranking Engine
Usage:
  python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
"""
import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Import modular components from toprank package
from toprank import (
    load_candidates,
    detect_honeypots,
    engineer_features,
    compute_final_score,
    generate_reasoning
)

def main():
    parser = argparse.ArgumentParser(description="Deterministic Candidate Ranking Pipeline")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl or candidates.jsonl.gz")
    parser.add_argument("--out", required=True, help="Path to output submission CSV file")
    args = parser.parse_args()

    print("Loading candidates dataset...")
    df = load_candidates(args.candidates)
    print(f"Loaded {len(df)} profiles.")

    print("Executing Anomaly Validation Firewall checks...")
    is_honeypot = detect_honeypots(df)
    num_honeypots = is_honeypot.sum()
    if num_honeypots > 0:
        print(f"Firewall flagged and removed {num_honeypots} anomalous candidates.")
        df = df[~is_honeypot].reset_index(drop=True)
    else:
        print("0 anomalies flagged.")

    print("Engineering candidate feature matrices...")
    features_df = engineer_features(df)
    
    script_dir = Path(__file__).resolve().parent
    embeddings_file = script_dir / "candidate_embeddings.npz"
    if embeddings_file.exists():
        print(f"Loading pre-computed similarities from {embeddings_file}...")
        data = np.load(embeddings_file, allow_pickle=True)
        sim_map = dict(zip(data['candidate_ids'], data['similarities']))
        similarities = df['candidate_id'].map(sim_map).fillna(0.5).values
    else:
        print("Warning: pre-computed embeddings file not found. Similarity calculations omitted.")
        similarities = np.full(len(df), 0.5)

    print("Computing final composite fit scores...")
    final_scores = compute_final_score(features_df, similarities)
    features_df['score'] = final_scores
    features_df['semantic_similarity'] = similarities

    # Optional BAAI/bge-reranker-v2-m3 Cross-Encoder Reranking for top 200 candidates
    try:
        from sentence_transformers import CrossEncoder
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # Try to load model locally first, then try hub, then fallback
            try:
                model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", local_files_only=True)
                print("Loaded cached BAAI/bge-reranker-v2-m3 cross-encoder.")
            except Exception as e_local:
                try:
                    print("Local cross-encoder not found. Attempting to load from Hugging Face Hub...")
                    model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", local_files_only=False)
                    print("Loaded BAAI/bge-reranker-v2-m3 cross-encoder from Hugging Face Hub.")
                except Exception as e_hub:
                    print(f"Failed to load cross-encoder from Hub: {e_hub}. Falling back to precomputed similarities.")
                    raise e_hub

        # Get top 200 candidate IDs based on baseline scoring
        top200_candidates = features_df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).head(200)
        top200_ids = top200_candidates['candidate_id'].tolist()

        # Build lookup for raw candidates
        raw_lookup = df[df['candidate_id'].isin(top200_ids)].set_index('candidate_id')

        # Build query-candidate pairs
        pairs = []
        c_ids = []
        JD_TEXT = """
        Senior AI Engineer role at a Series A AI startup. We need deep technical depth in modern ML systems—embeddings, retrieval, ranking, LLMs, fine-tuning, and a scrappy product-engineering attitude.
        Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5) deployed to real users.
        Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
        Strong Python. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).
        """
        for cid in top200_ids:
            raw_cand = raw_lookup.loc[cid]
            profile = raw_cand.get('profile', {}) or {}
            summary = profile.get('summary', '')
            summary_words = summary.split()[:35]
            parts = [
                profile.get('headline', ''),
                ' '.join(summary_words),
                profile.get('current_title', ''),
            ]
            for job in raw_cand.get('career_history', []):
                if isinstance(job, dict):
                    parts.append(job.get('title', ''))
            cand_text = ' '.join(filter(None, parts)).lower()
            pairs.append([JD_TEXT, cand_text])
            c_ids.append(cid)

        print(f"Reranking {len(pairs)} candidates with BAAI/bge-reranker-v2-m3 on CPU...")
        ce_scores = model.predict(pairs, batch_size=32)

        # Min-max scale the cross-encoder scores to [0.4, 0.95] to blend with semantic similarity scale
        ce_min = ce_scores.min()
        ce_max = ce_scores.max()
        if ce_max > ce_min:
            scaled_scores = 0.4 + 0.55 * (ce_scores - ce_min) / (ce_max - ce_min)
        else:
            scaled_scores = np.full(len(ce_scores), 0.7)

        # Update similarities in features_df
        ce_map = dict(zip(c_ids, scaled_scores))
        for idx, row in features_df.iterrows():
            cid = row['candidate_id']
            if cid in ce_map:
                similarities[idx] = ce_map[cid]

        print("Recomputing final scores with Cross-Encoder similarities...")
        final_scores = compute_final_score(features_df, similarities)
        features_df['score'] = final_scores
        features_df['semantic_similarity'] = similarities

    except Exception as e:
        print(f"Cross-Encoder reranking skipped/unavailable ({type(e).__name__}). Using pre-computed similarities.")

    print("Ranking and resolving ties deterministically...")
    features_df = features_df.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    
    top100 = features_df.head(100).copy()
    
    s_min = top100['score'].min()
    s_max = top100['score'].max()
    if s_max > s_min:
        top100['score'] = (top100['score'] - s_min) / (s_max - s_min)
    else:
        top100['score'] = 1.0
        
    top100['score'] = top100['score'].round(4)
    top100 = top100.sort_values(by=['score', 'candidate_id'], ascending=[False, True]).reset_index(drop=True)
    top100['rank'] = range(1, len(top100) + 1)

    # Merge raw candidate data back for rich reasoning generation
    top100_ids = set(top100['candidate_id'].tolist())
    raw_lookup = df[df['candidate_id'].isin(top100_ids)].set_index('candidate_id')
    for col in ['profile', 'skills', 'career_history', 'education', 'redrob_signals']:
        if col in raw_lookup.columns:
            top100[col] = top100['candidate_id'].map(raw_lookup[col])

    print("Generating fact-based reasonings...")
    top100['reasoning'] = top100.apply(generate_reasoning, axis=1)
    
    submission_df = top100[['candidate_id', 'rank', 'score', 'reasoning']]
    
    print(f"Saving final ranking CSV to {args.out}...")
    submission_df.to_csv(args.out, index=False)
    print("Ranking complete. Submission file saved successfully.")

if __name__ == '__main__':
    main()
