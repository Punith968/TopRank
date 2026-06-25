import gradio as gr
import subprocess
import os
import sys
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))


def execute_pipeline(input_file, progress=gr.Progress()):
    """Execute the TopRank ranking pipeline end-to-end."""
    if not input_file:
        raise gr.Error("Please upload a candidates JSONL file first.")

    input_path = input_file.name if hasattr(input_file, 'name') else str(input_file)
    output_csv = os.path.join(APP_DIR, "submission.csv")
    rank_script = os.path.join(APP_DIR, "rank.py")

    cmd = [sys.executable, rank_script,
           "--candidates", input_path,
           "--out", output_csv]

    try:
        progress(0.1, desc="Loading candidates & running firewall...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True,
            cwd=APP_DIR, timeout=300
        )
        progress(0.9, desc="Preparing results...")
    except subprocess.TimeoutExpired:
        raise gr.Error("Pipeline timed out (5-minute limit exceeded).")
    except subprocess.CalledProcessError as e:
        error_msg = (e.stderr or e.stdout or "Unknown error")[:600]
        raise gr.Error(f"Pipeline failed:\n{error_msg}")

    if not os.path.exists(output_csv):
        raise gr.Error("Pipeline completed but submission.csv was not found.")

    # Build rich preview
    df = pd.read_csv(output_csv)
    top_rows = []
    for _, r in df.head(10).iterrows():
        reason = str(r['reasoning'])
        short = (reason[:100] + '...') if len(reason) > 100 else reason
        top_rows.append(
            f"| {int(r['rank'])} | `{r['candidate_id']}` | {r['score']:.4f} | {short} |"
        )

    preview = "### \U0001f3c6 Top 10 Candidates\n\n"
    preview += "| Rank | Candidate ID | Score | Reasoning |\n"
    preview += "|---:|:---|---:|:---|\n"
    preview += "\n".join(top_rows)
    preview += f"\n\n*Showing 10 of {len(df)} ranked candidates*"

    progress(1.0, desc="Done!")
    log_text = result.stdout or "Pipeline completed successfully."
    status = f"✅ Ranked {len(df)} candidates successfully"

    # Copy output to system temp directory to ensure Gradio's sandbox allows downloading it
    import shutil
    import tempfile
    temp_dir = tempfile.gettempdir()
    temp_csv = os.path.join(temp_dir, "submission.csv")
    shutil.copy2(output_csv, temp_csv)

    return temp_csv, preview, status, log_text


# ── Theme & CSS ──────────────────────────────────────────────────────────
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="emerald",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)

css = """
.main-hdr{text-align:center;padding:24px 0 4px}
.main-hdr h1{font-size:2.6em;background:linear-gradient(135deg,#667eea,#764ba2);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0}
.sub{text-align:center;color:#777;font-size:1.05em;margin:0 0 8px}
.foot{text-align:center;color:#aaa;font-size:.82em;padding:14px}
"""

# ── Layout ───────────────────────────────────────────────────────────────
with gr.Blocks(title="TopRank \u2014 Candidate Ranking Engine",
               theme=theme, css=css) as demo:

    gr.HTML('<div class="main-hdr"><h1>\U0001f680 TopRank</h1></div>')
    gr.Markdown(
        "**Intelligent Candidate Ranking Engine** \u2014 Upload a `candidates.jsonl` "
        "file, run the deterministic multi-stage pipeline, and download the "
        "ranked `submission.csv`.",
        elem_classes="sub",
    )
    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("#### \U0001f4c1 Input")
            file_in = gr.File(
                label="Upload Candidates (JSONL)",
                file_types=[".jsonl", ".json", ".gz"],
            )
            btn = gr.Button(
                "\U0001f3c3 Run Ranking Pipeline",
                variant="primary", size="lg",
            )
            status = gr.Textbox(label="Status", interactive=False, lines=1)
            file_out = gr.File(
                label="\U0001f4e5 Download Submission CSV",
                interactive=False,
            )

        with gr.Column(scale=2):
            gr.Markdown("#### \U0001f4ca Results Preview")
            preview = gr.Markdown(
                value="*Upload a file and click **Run** to see results here.*"
            )

    with gr.Accordion("\U0001f4cb Pipeline Logs", open=False):
        logs = gr.Textbox(
            label="Output", interactive=False, lines=8, max_lines=20,
        )

    gr.Markdown("---")
    gr.HTML(
        '<div class="foot">'
        'TopRank v1.0 \u00b7 Deterministic Multi-Stage Candidate Ranking \u00b7 '
        '<a href="https://github.com/Punith968/TopRank" target="_blank">GitHub</a> \u00b7 '
        '<a href="https://huggingface.co/spaces/Punith3068/TopRank" target="_blank">HF Space</a>'
        '</div>'
    )

    btn.click(
        fn=execute_pipeline,
        inputs=file_in,
        outputs=[file_out, preview, status, logs],
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        allowed_paths=[APP_DIR]
    )
