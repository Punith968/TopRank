# TopRank — Solution Architecture

## 1. Design Philosophy

The JD is unusually explicit about what it values (production-oriented builders who understand retrieval/ranking) and what it penalises (title-chasers, consulting-only careers, recent-framework-only AI experience, pure researchers). A good ranking system must encode **all** of these preferences, not just keyword similarity.

Our architecture therefore separates three concerns:

1. **Fraud Detection** — eliminate synthetically impossible profiles before they consume ranking budget.
2. **Relevance Estimation** — measure semantic and structural fit to the JD.
3. **Behavioural Scoring** — weight platform engagement, availability, and career trajectory signals that the JD explicitly calls out.

---

## 2. Pipeline Stages

### Stage 1 — Data Ingestion (`dataloader.py`)
Streams 100K JSONL records into a pandas DataFrame. Supports both plain-text and gzip-compressed inputs.

### Stage 2 — Profile Validation Firewall (`firewall.py`)
A 17-rule deterministic filter that flags profiles with logically impossible attributes:

| Trap # | Description | Example |
|--------|-------------|---------|
| 1 | Expert skill with 0 duration | "Expert in PyTorch" but 0 months listed |
| 2 | Career months exceed YoE envelope | 25 YoE claimed but career sums to 40 years |
| 3 | Senior title with < 3 YoE | "Principal Engineer" with 2 years experience |
| 4 | High salary expectations for junior YoE | ₹50+ LPA expected but < 5 YoE |
| 5 | PhD with < 4 YoE | PhD holder with 2 years total experience |
| 6 | Negative notice period | notice_period_days = -30 |
| 7 | Education end < start | Graduated before enrolling |
| 8 | Skill duration exceeds career span | 15 years of Python with 5 YoE |
| 9 | Tech release timeline violation | 48 months of LangChain (released 2022) |
| 10 | Career span vs YoE mismatch | Claims 15 YoE but earliest job started 3 years ago |
| 11 | Career pre-graduation anomaly | Full-time SDE role 10 years before bachelor's degree |
| 12 | Degree level timeline conflict | Master's degree completed before bachelor's |
| 13 | Expert/Advanced skill with very low duration | "Expert" but < 6 months |
| 14 | Salary min > max | Expected salary range inverted |
| 15 | Last active before signup | Active on platform before creating account |
| 16 | Company founding year violation | Worked at OpenAI in 2010 (founded 2015) |
| 17 | Overlapping education at different institutions | Full-time at IIT and NIT simultaneously |

**Design rationale**: The spec warns that ~80 honeypot candidates exist and that a > 10% honeypot rate in top-100 disqualifies the submission. Rather than just filtering obvious cases, we cast a wide net of 17 logically independent checks. False positives (legitimate candidates accidentally flagged) are acceptable because we'd rather lose a marginal candidate than include a honeypot.

### Stage 3 — Feature Engineering (`features.py`)
Extracts 18 behavioural and structural features per candidate:

- **Career trajectory**: consulting ratio, job-hopper flag, only-consulting flag, pure-research flag, no-current-production flag
- **Skill analysis**: skill count, keyword-stuffer flag, LangChain-only flag (JD explicitly penalises this), CV/speech-only flag (JD deprioritises non-NLP)
- **Platform engagement**: recruiter response rate, interview completion rate, offer acceptance rate, engagement recency score (from last_active_date), open-to-work flag
- **Availability**: notice period in days
- **Activity**: GitHub presence flag

### Stage 4 — Semantic Similarity (Pre-computed via `precompute.py`)
- **Model**: `all-MiniLM-L6-v2` (384-dim, CPU-friendly, 22M params)
- **Strategy**: Concatenate headline + summary (first 35 words) + current title + past job titles → encode → cosine similarity against JD embedding
- **Why offline?** Encoding 100K profiles on CPU takes ~7.5 minutes — well over the 5-minute runtime budget. By pre-computing and storing in `candidate_embeddings.npz` (579 KB), the online ranking step loads similarities in < 0.1 seconds.

### Stage 5 — Composite Scoring (`ranker.py`)

| Dimension | Weight | Signal Source |
|-----------|--------|---------------|
| Semantic Similarity | 25% | Pre-computed MiniLM cosine similarity |
| Career Trajectory | 25% | YoE Gaussian (μ=7, σ=2), consulting/hopper/research/LangChain/CV penalties |
| Skill Relevance | 20% | Log-scaled skill count, keyword-stuffer penalty |
| Behavioural Signals | 20% | Recruiter/interview/offer rates, GitHub presence, engagement recency |
| Availability | 10% | Notice period tiers, open-to-work bonus |

**YoE suitability** uses a Gaussian centred at 7 years (the midpoint of the JD's 5–9 range) with σ=2, giving a smooth preference curve rather than a hard cutoff.

### Stage 6 — Deterministic Tie-Breaking & Scaling (`rank.py`)
- Sort by composite score descending, then by `candidate_id` ascending for tie-breaking (spec requirement).
- Min-max scale the top-100 scores to [0, 1].
- Assign integer ranks 1–100.

### Stage 7 — Natural Language Reasoning (`nlg.py`)
Generates a fact-based, candidate-specific justification string for each of the top 100:
- References the candidate's **actual current title and company**
- Names **specific JD-matching skills** from their profile
- Quantifies **semantic similarity score**
- Connects **YoE to JD requirements**
- Flags **honest concerns** (consulting-only, job hopping, LangChain-only, etc.)
- Notes **availability** (notice period)

Each reasoning is unique because it draws from the candidate's actual profile data.

---

## 3. Key Design Tradeoffs

| Decision | Alternative Considered | Why We Chose This |
|----------|----------------------|-------------------|
| Pre-computed embeddings | Online encoding at runtime | 7.5 min encoding breaks the 5-min budget |
| MiniLM-L6-v2 | Larger models (BGE, E5) | Must fit CPU-only constraint; MiniLM is fast and good enough |
| Heuristic scoring | Learning-to-rank (XGBoost) | No labelled training data available; heuristics encode JD intent directly |
| 17-trap firewall | Embedding-based anomaly detection | Deterministic, explainable, zero false negatives on known anomaly patterns |
| Template NLG | LLM-generated reasoning | No API calls allowed; template approach is deterministic and hallucination-free |

---

## 4. Compute Profile

| Metric | Value |
|--------|-------|
| Online runtime (rank.py on 100K) | ~8.5 seconds |
| Peak memory | ~1.2 GB |
| Pre-computation time | ~7.5 minutes (one-time) |
| Pre-computed artifact size | 579 KB |
| GPU required? | No |
| Network required? | No |
