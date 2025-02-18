# Neural Circuits

Code to accompany the paper "A Metalearned Neural Circuit for Nonparametric Bayesian Inference" (NeurIPS 2024).

Our code consists of two packages:

1. `partikel`, which contains code for running particle filters.
2. `antelope`, which contains code for neural circuits and running experiments.

## Installation

1. Create a conda environment:

    ```
    conda create --name circuit python=3.9
    ```

1. Activate the environment:

    ```
    conda activate circuit
    ```

1. Install the `partikel` package (for particle filtering):

    ```
    cd partikel
    pip install -r requirements.txt
    pip install -e .
    ```

1. Run the `partikel` tests:

    ```
    pytest
    ```

    They should pass.

1. `cd` to the `antelope` directory, which contains the neural circuit code:

    ```
    cd ../antelope
    ```

1. Install the requirements:

    ```
    pip install -r requirements.txt
    ```

1. Install `antelope`:

    ```
    pip install -e .
    ```
    
1. Run `antelope` tests:

    ```
    pytest
    ```

    The tests should all pass. The environment is now ready. Henceforth, all commands should be run from the `antelope` directory.

## Generating Synthetic Data

Please note that this step requires about 183MB of disk space. From the `antelope` directoy, run the following command:

```
python scripts/generate_synthetic_data.py
```

This will create a file `synthetic_normal_inverse_gamma_data.npz` that can be used for training a neural circuit. The data generating process can be configured by modifying the variables: `alpha` (CRP coefficient), `dim` (dimensionality of data), `loc` (normal inverse gamma $m$ parameter), `mean_conc` (normal inverses gamma $\lambda$ parameter), `conc` (normal inverse gamma $a$ parameter), and `scale` (normal inverse gamma $b$ parameter).

## Training a Neural Circuit

We can now use the generated data to train a neural circuit. First create a directory to store checkpoints:

```
mkdir checkpoints
```

Then run the following command to start training (pass `--device cuda` to run on the gpu). You can also change `max_iter` to a lower value if necessary, though this may affect performance of the neural circuit.

```
python scripts/train_circuit.py --data_file synthetic_normal_inverse_gamma_data.npz \
    --max_iter 10000 --out_dir checkpoints --batch_size 128 --hidden_size 1024 \
    --num_layers 2 --gamma 1.0 --job_id demo-circuit
```

## Evaluating a Neural Circuit

The circuit can be evaluated by running the following command:

```
python scripts/eval_generic.py --data_file synthetic_normal_inverse_gamma_data.npz \
    checkpoints/demo-circuit/circuit_model.pt
```

The evaluation results will be placed directly in the checkpoint directory.

## Evaluating a Particle Filter

In the synthetic case, the true hyperparameters are known and so no training is necessary. Create an empty checkpoint directory:

```
mkdir checkpoints/synthetic_normal_inverse_gamma_data_expfamily
```

Then evaluate it (note that the true hyperparameters are set directly in the evaluation script):

```
python scripts/eval_generic.py --data_file synthetic_normal_inverse_gamma_data.npz \
    checkpoints/synthetic_normal_inverse_gamma_data_expfamily
```

## Summarizing Evaluation Results

The evaluation results can be summarized by running the following command:

```
python scripts/summarize_results.py synthetic_normal_inverse_gamma_data_expfamily demo-circuit
```

