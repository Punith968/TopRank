# toprank Architecture

This document describes the high-performance candidate screening and ranking system, **toprank**.

## Key Pipelines

### 1. Data Ingestion
Loads candidates streaming from `data/candidates.jsonl` using python `json` and `gzip`.

### 2. Profile Validation Firewall (17 stages)
Checks 17 distinct logical rules to eliminate impossible profile submissions.

### 3. Feature Ingestion & Normalization
Computes career trajectory features, job-hopper scores, and consulting ratios.

### 4. Semantic Similarity
Loads pre-computed MiniLM cosine similarities to represent profile-to-JD relevance.

### 5. Scoring & scaling
Applies weighted combination of features and scales the leaderboard scores to [0,1].
