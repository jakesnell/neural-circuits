import click
import os

import numpy as np

import pandas as pd


@click.command()
@click.argument("job_ids", nargs=-1)
@click.option("--checkpoint_dir", default=os.path.join(os.getcwd(), "checkpoints"))
def main(job_ids, checkpoint_dir):
    records = []
    for job_id in job_ids:
        job_dir = os.path.join(checkpoint_dir, job_id)
        val_nll_trace = np.load(os.path.join(job_dir, "val_nll_trace.npy"))
        val_clustering = np.load(os.path.join(job_dir, "val_clustering.npz"))
        test_nll_trace = np.load(os.path.join(job_dir, "test_nll_trace.npy"))
        test_benchmark = np.load(os.path.join(job_dir, "test_benchmark.npz"))
        test_clustering = np.load(os.path.join(job_dir, "test_clustering.npz"))

        record = {
            "job_id": job_id,
            "val_mean_nll": val_nll_trace.mean().item(),
            "val_perplexity": np.exp(val_nll_trace.mean(-1)).mean(0).item(),
            "val_ari": val_clustering["adjusted_rand"].mean(0).item(),
            "val_mi": val_clustering["adjusted_mutual_info"].mean(0).item(),
            "test_mean_nll": test_nll_trace.mean().item(),
            "test_perplexity": np.exp(test_nll_trace.mean(-1)).mean(0).item(),
            "test_ari": test_clustering["adjusted_rand"].mean(0).item(),
            "test_mi": test_clustering["adjusted_mutual_info"].mean(0).item(),
            "nll_time_per_example_ms": 1000
            * test_benchmark["elapsed"]
            / test_benchmark["num_examples"],
            "map_time_per_example_ms": 1000
            * test_clustering["map_predict_time"]
            / test_clustering["num_examples"],
        }

        records.append(record)

    df = pd.DataFrame(records)
    df.set_index("job_id", inplace=True)
    print(df)


if __name__ == "__main__":
    main()
