import click
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def moving_average(x: torch.Tensor, window_size: int):
    x_padded = F.pad(x.unsqueeze(0), (window_size - 1, 0), mode="replicate").squeeze(0)
    return x_padded.unfold(dimension=0, size=window_size, step=1).mean(-1)


@click.command()
@click.argument("job_ids", nargs=-1)
@click.option("--checkpoint_dir", default=os.path.join(os.getcwd(), "checkpoints"))
@click.option("--ymax", default=0.0)
@click.option("--window_size", default=50)
@click.option("--outfile", default="")
def main(job_ids, ymax, checkpoint_dir, window_size, outfile):
    for job_id in job_ids:
        losses = torch.load(os.path.join(checkpoint_dir, job_id, "loss_vals.pt"))
        print(
            f"{job_id}: final loss = {losses[-1]:0.4f}, best loss = {losses.min():0.4f}"
        )
        plt.plot(moving_average(losses, window_size), label=job_id)

    if ymax > 0.0:
        plt.ylim((0, ymax))

    plt.legend(loc="best")
    if outfile == "":
        plt.show()
    else:
        plt.savefig(outfile, dpi=300)


if __name__ == "__main__":
    main()
