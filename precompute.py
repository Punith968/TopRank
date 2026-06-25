#!/usr/bin/env python3
"""
toprank Offline Embedding Pre-computation Script
"""
import argparse
import gzip
import json
import os
import sys
import time
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print("Error: sentence-transformers and scikit-learn are required to run this script.")
    sys.exit(1)

JD_TEXT = """
Senior AI Engineer role at a Series A AI startup. We need deep technical depth in modern ML systems—embeddings, retrieval, ranking, LLMs, fine-tuning, and a scrappy product-engineering attitude.
Production experience with embeddings-based retrieval systems (sentence-transformers, BGE, E5) deployed to real users.
Production experience with vector databases or hybrid search infrastructure (Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS).
Strong Python. Hands-on experience designing evaluation frameworks for ranking systems (NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation).
"""

MODEL_NAME = 'all-MiniLM-L6-v2'

def main():
    parser = argparse.ArgumentParser(description="Precompute candidate embeddings and similarities")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name to use")
    parser.add_argument("--out", default="candidate_embeddings.npz", help="Output path for embeddings/similarities file")
    args = parser.parse_args()

    start_time = time.time()
    print("Loading candidate profiles...")
    candidate_ids = []
    texts = []
    
    open_func = gzip.open if args.candidates.endswith('.gz') else open
    with open_func(args.candidates, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            cand = json.loads(line)
            candidate_ids.append(cand['candidate_id'])
            
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
            
    print(f"Loaded {len(candidate_ids)} candidates.")
    print(f"Loading model '{args.model}'...")
    model = SentenceTransformer(args.model)
    
    print("Embedding candidates...")
    candidate_embeddings = model.encode(texts, batch_size=512, show_progress_bar=True, convert_to_numpy=True)
    jd_embedding = model.encode([JD_TEXT], convert_to_numpy=True)
    
    print("Computing similarities...")
    similarities = cosine_similarity(candidate_embeddings, jd_embedding).flatten()
    
    np.savez_compressed(args.out, candidate_ids=candidate_ids, similarities=similarities)
    print(f"Finished in {time.time() - start_time:.2f}s.")

if __name__ == '__main__':
    main()
