import os
import csv 
import pandas as pd
import matplotlib.pyplot as plt

# Make log path relative to project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Goes up from src/ to root
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(LOG_DIR, exist_ok=True)  # Create logs/ if missing
log_path = os.path.join(LOG_DIR, "metrics.csv")  

def log_metrics(query, metrics):
    
    file_exists = os.path.isfile(log_path) # Check if log file exists
    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "latency_sec", "num_chunks", "avg_score", "answer"]) # Define columns
        if not file_exists:
            writer.writeheader()
        row = {"query": query, **metrics}
        del row["retrieved_chunks"]  # to avoid logging full chunks
        writer.writerow(row)
    print(f"Logged metrics for: {query}")  # Debug feedback

def visualize_metrics():
    if not os.path.isfile(log_path):
        print("No metrics file yet.")
        return
    df = pd.read_csv(log_path)
    if df.empty or 'query' not in df.columns:
        print("No data or 'query' column missing in metrics.csv - skipping plot.")
        return
    df.plot(x="query", y=["latency_sec", "avg_score"], kind="bar")
    plt.title("Query Metrics Overview")
    plt.savefig("metrics_plot.png")
    print("Saved metrics_plot.png")