
import pandas as pd
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def aggregate_results(input_path, output_path):
    """
    Read benchmark summary CSV and calculate average metrics for each dataset.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    logger.info(f"Reading results from {input_path}...")
    df = pd.read_csv(input_path)

    # Group by dataset and calculate mean of numeric columns
    numeric_cols = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    # Filter for columns that actually exist in the dataframe
    numeric_cols = [col for col in numeric_cols if col in df.columns]

    logger.info("Aggregating results...")
    # We use 'dataset' as the grouping key. 
    # For single-target datasets (e.g. BBBP), the average is the same as the single row.
    # For multi-target (e.g. Tox21), it provides the mean across all targets.
    summary_avg = df.groupby('dataset')[numeric_cols].mean().reset_index()

    # Add a count column to show how many targets were aggregated
    counts = df.groupby('dataset').size().reset_index(name='target_count')
    summary_avg = pd.merge(summary_avg, counts, on='dataset')

    # Format numeric columns for better readability (optional, but good for CSV)
    # Keeping them as floats is better for further processing, but rounding makes it readable.
    # Let's keep full precision in CSV.

    logger.info(f"Saving aggregated results to {output_path}...")
    summary_avg.to_csv(output_path, index=False)
    
    # Print to console
    print("\nAggregated Benchmark Results:")
    print(summary_avg.to_string(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate Benchmark Results")
    parser.add_argument('--input', type=str, default='output/benchmark/benchmark_summary.csv', help='Input CSV path')
    parser.add_argument('--output', type=str, default='output/benchmark/benchmark_summary_averaged.csv', help='Output CSV path')
    args = parser.parse_args()

    aggregate_results(args.input, args.output)
