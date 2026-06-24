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
