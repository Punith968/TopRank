import gradio as gr
import json
import pandas as pd
import os

def load_data():
    try:
        path = "data/sample_candidates.json"
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def get_candidates_df(yoe_min, yoe_max, search_query):
    candidates = load_data()
    rows = []
    for cand in candidates:
        prof = cand.get('profile', {}) or {}
        yoe = prof.get('years_of_experience', 0) or 0
        title = prof.get('current_title', 'Software Engineer')
        skills = ", ".join([s.get('name', '') for s in cand.get('skills', [])[:5] if isinstance(s, dict)])
        
        if not (yoe_min <= yoe <= yoe_max):
            continue
            
        if search_query:
            q = search_query.lower()
            if q not in title.lower() and q not in skills.lower():
                continue
            
        rows.append({
            "ID": cand.get('candidate_id'),
            "Current Title": title,
            "Years of Experience": yoe,
            "Key Skills": skills
        })
        
    return pd.DataFrame(rows)

demo = gr.Interface(
    fn=get_candidates_df,
    inputs=[
        gr.Slider(0, 30, value=5, label="Min Years of Experience"),
        gr.Slider(0, 30, value=15, label="Max Years of Experience"),
        gr.Textbox(label="Search Title / Skills")
    ],
    outputs=gr.Dataframe(label="Candidates Leaderboard"),
    title="toprank Candidate Screening Dashboard",
    description="Interactive exploration of screened and ranked candidates."
)

if __name__ == "__main__":
    demo.launch(server_port=7860)
