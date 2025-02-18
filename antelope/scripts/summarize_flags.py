import click
import os

import torch

import pandas as pd

blacklist = [
    "logtostderr",
    "alsologtostderr",
    "log_dir",
    "v",
    "verbosity",
    "logger_levels",
    "stderrthreshold",
    "showprefixforinfo",
    "run_with_pdb",
    "pdb_post_mortem",
    "pdb",
    "run_with_profiling",
    "profile_file",
    "use_cprofile_for_profiling",
    "only_check_args",
    "seed",
    "plot",
    "debug",
    "out_dir",
    "?",
    "help",
    "helpshort",
    "helpfull",
    "helpxml",
]


@click.command()
@click.argument("job_ids", nargs=-1)
@click.option("--checkpoint_dir", default=os.path.join(os.getcwd(), "checkpoints"))
@click.option("--exclude", default="data_file,data_type,device,compile")
def main(job_ids, checkpoint_dir, exclude):
    records = []
    for job_id in job_ids:
        record = torch.load(os.path.join(checkpoint_dir, job_id, "flags.pt"))
        for flag in blacklist:
            if flag in record:
                del record[flag]
        for flag in exclude.split(","):
            if flag in record:
                del record[flag]
        records.append(record)

    df = pd.DataFrame(records)
    df.set_index("job_id", inplace=True)
    print(df)


if __name__ == "__main__":
    main()
