import gradio as gr
import subprocess
import os

def execute_pipeline(input_file):
    if not input_file:
        return None
    
    # Define output path in the current working directory
    output_csv = os.path.abspath("submission.csv")
    
    # Execute the command-line ranking script just like the judges will
    cmd = [
        "python", "rank.py",
        "--candidates", input_file.name,
        "--out", output_csv
    ]
    
    try:
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running rank.py:")
        print(e.stderr)
        raise gr.Error(f"Ranking pipeline failed: {e.stderr}")
        
    if os.path.exists(output_csv):
        return output_csv
    else:
        raise gr.Error("Pipeline finished but output CSV was not found.")

with gr.Blocks(title="TopRank Sandbox") as demo:
    gr.Markdown("# TopRank Pipeline Sandbox")
    gr.Markdown("Upload a `candidates.jsonl` file to run the deterministic ranking pipeline end-to-end and generate the `submission.csv` file, fulfilling the Sandbox Requirement (Section 10.5).")
    
    with gr.Row():
        file_in = gr.File(label="Upload Candidates (JSONL)")
        file_out = gr.File(label="Download Submission (CSV)", interactive=False)
        
    btn = gr.Button("Run End-to-End Ranking", variant="primary")
    
    btn.click(
        fn=execute_pipeline,
        inputs=file_in,
        outputs=file_out
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
