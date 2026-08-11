# Stochastic Flow Map Learning with Normalizing Flows

This repository contains normalizing-flow implementations for stochastic flow map learning. The code supports three related model forms:

- **NF**: a direct normalizing-flow model for stochastic flow-map transitions, $x_{n+1} = G_\theta(x_n, z)$.
- **ResNF**: a residual normalizing-flow model, $x_{n+1} = x_n + G_\theta(x_n, z)$.
- **MixNF**: a mixed normalizing-flow model, $x_{n+1} = \Phi(x_n) + G_\theta(x_n, z)$. In the SSA examples from the second SSA paper, $\Phi$ is an ODE or chemical-dynamics prior, and the normalizing flow learns the remaining stochastic component around that prior.

The examples in this repository apply these models to three settings: ordinary stochastic differential equations, including autonomous and nonautonomous systems; Markovian effective dynamics of multiscale SDEs; and stochastic simulation algorithm models for chemical reaction networks.

## References

The included examples are associated with the following papers:

- [1] Yuan Chen and Dongbin Xiu, `Modeling Unknown Stochastic Dynamical System Subject to External Excitation`, 2026.
  https://iamyuanchen.xyz/pdf/2026ChenXiu.pdf
- [2] Yuan Chen and Dongbin Xiu, `Data-Driven Effective Modeling of Multiscale Stochastic Dynamical Systems`, 2024.
  https://iamyuanchen.xyz/pdf/2024ChenXiu_b.pdf
- [3] Yuan Chen, Weize Mao, and Dongbin Xiu, `Data-Driven Effective Modeling of Stochastic Chemical Reaction Networks`, to be published soon.
- [4] Yuan Chen, Markos A. Katsoulakis, and Dongbin Xiu, `Chemistry-Informed Generative Modeling for Complex Stochastic Reaction Networks`, to be published soon.

## Requirements

The code was developed with Python and PyTorch. A working environment should include:

- Python 3
- PyTorch
- NumPy
- SciPy
- Matplotlib
- scikit-learn
- munch
- absl-py
- tqdm

## Repository Structure

The main folders are:

| Folder | Contents |
| --- | --- |
| `config/` | Example JSON configs. The current cleaned copy keeps a small representative config for the autonomous SDE example. |
| `data/` | MATLAB `.mat` datasets. The current cleaned copy includes selected small data files for the Ex4 autonomous SDE example. |
| `results/` | Cleaned result folders for the included paper examples. Each folder keeps config and trained weights only. |
| `src/` | Supporting source files used by the flow models. |

The main files are:

| File | Purpose |
| --- | --- |
| `SolveNFSDE.py` | Training entry point for ordinary NF SDE models, including autonomous SDE examples and some SSA examples. |
| `SolveResNFSDE.py` | Training entry point for residual NF SDE models. |
| `SolveMixNFSDE.py` | Training entry point for mixed NF models. |
| `SolveNFNonAutoSDE.py` | Training entry point for nonautonomous NF SDE models. |
| `SolveResNFNonAutoSDE.py` | Training entry point for residual nonautonomous NF SDE models. |
| `SolveMixNFNonAutoSDE.py` | Training entry point for mixed nonautonomous NF SDE models. |
| `NFSDE.py` | Core NF SDE model implementation. |
| `ResNFSDE.py` | Core residual NF SDE model implementation. |
| `MixNFSDE.py` | Core mixed NF SDE model implementation. |
| `NFNonAutoSDE.py` | Nonautonomous NF model implementation. |
| `ResNFNonAutoSDE.py` | Nonautonomous residual NF model implementation. |
| `MixNFNonAutoSDE.py` | Nonautonomous mixed NF model implementation. |
| `NFSDE_SSAconserve.py` | Conservative NF variant for SSA examples. |
| `ResNFSDE_SSAgenconserve.py` | Residual/conservative NF variant for SSA examples. |
| `MixNFSDE_SSAgenconserve.py` | Mixed/conservative NF variant for SSA examples. |
| `Chemical_Dynamics.py` | Chemical-dynamics helper functions used by SSA examples. |
| `ShowTest.py` | Testing and prediction entry point for trained models. |
| `ShowProdcution.py` | Postprocessing entry point for production/paper-style figures. |
| `ShowPerformance.py` | Additional performance and diagnostic plotting utility. |
| `Evaulation.py` and `Prodcution.py` | Evaluation and production plotting utilities. |

## Included Example

The cleaned `results/` folder contains trained configs and weights for the following paper examples.

### Nonautonomous and Controlled SDEs

These examples correspond to [1].

| Example | Result folder | Description | Model type | Provided |
| --- | --- | --- | --- | --- |
| Ex12 | `results/Ex12` | OU process with drift control | NF | Config, final weights, ensemble weights |
| Ex15 | `results/Ex15` | Stochastic resonance / double-well with excitation | NF | Config and final weights |
| Ex16 | `results/Ex16` | Nonlinear SDE with control | NF | Config and final weights |
| Ex17 | `results/Ex17` | Stochastic predator-prey model with excitation | NF | Config and final weights |
| Ex19 | `results/Ex19` | OU process with both drift and diffusion control | NF | Config, final weights, ensemble weights |
| Ex43 | `results/Ex43` | Gene expression SSA model with time-dependent reaction rate | ResNF | Config and final weights |
| SPDEEx3 | `results/SPDEEx3` | Stochastic heat equation with source, modal/spectral form | ResNF | Config and final weights |

### Markovian Effective Dynamics of Multiscale SDEs

These examples correspond to [2].

| Example | Result folder | Description | Model type | Provided |
| --- | --- | --- | --- | --- |
| Ex28 | `results/Ex28` | Skew product SDE | NF | Config and final weights |
| Ex33 | `results/Ex33` | Exponential mean OU / multiscale exponential example | NF | Config and final weights |
| Ex38 | `results/Ex38` | Triad system | NF | Config and final weights |
| Ex34 | `results/Ex34` | 3D nonlinear multiscale SDE | NF | Config and final weights |
| Ex36 | `results/Ex36` | Multiscale stochastic oscillator | NF | Config and final weights |

### SSA Paper I

These examples correspond to [3].

| Example | Result folder | Description | Model type | Provided |
| --- | --- | --- | --- | --- |
| Ex22 | `results/Ex22` | Transfer process | NF / conservative NF | Config and final weights |
| Ex23 LV slow | `results/Ex23_LVSlow` | Slow Lotka-Volterra SSA model | NF | Config and final weights |
| Ex23 LV fast | `results/Ex23_LVFast` | Fast Lotka-Volterra SSA model | ResNF | Config and final weights |
| Ex25 | `results/Ex25` | Brusselator | NF | Config and final weights |
| Ex27 | `results/Ex27` | Autocatalysis | ResNF | Config and final weights |
| Ex26 | `results/Ex26` | Oregonator | ResNF | Config and final weights |

### SSA Paper II

These examples correspond to [4].

| Example | Result folder | Description | Model type | Provided |
| --- | --- | --- | --- | --- |
| Ex45 | `results/Ex45` | Schlogl model | MixNF | Config, final weights, best weights |
| Ex42 | `results/Ex42` | Vilar 2002 genetic oscillator model | MixNF / conservative MixNF | Config, final weights, best weights |
| Ex41 | `results/Ex41` | Mammalian circadian clock model | MixNF | Config, final weights, best weights |
| Ex23 Mix | `results/Ex23_Mix` | Lotka-Volterra SSA model with mixed prior | MixNF | Config and final weights |

## Preparation

### Config

Each run is controlled by a config file. For cleaned result folders, `results/<example>/Test_config.json` is the record of the settings used for that trained model when the config is available.

The main config sections follow the same pattern across model classes:

`eqn_config`: equation or reaction-network settings.

- `eqn_name`: example or equation name.
- `dim`: dimension of the state variable when applicable.
- `Delta`: time-step size when applicable.
- Example-specific parameters: drift, diffusion, reaction-rate, control, or external-excitation parameters.
- Prior/residual settings: for ResNF or MixNF examples, this may include the deterministic/residual prior model type and paths.

`net_config`: normalizing-flow architecture and training settings.

- `fname`: flow architecture. The provided examples mainly use masked autoregressive flows (`MAF`).
- Hidden-layer, node, optimizer, learning-rate, batch-size, and epoch settings.
- Model-specific settings for NF, ResNF, MixNF, conservative variants, and nonautonomous variants.

`dat_config`: data paths and sampling settings.

- `TrainData_dir`: path to the training `.mat` file.
- `TestData_dir`: path to the test `.mat` file.
- Additional data paths for conditional SSA data, nonautonomous parameters, or production tests when used.

`show_config` and `monitor_config`: postprocessing and monitoring settings.

- These sections control sample plots, mean/std plots, density plots, loss plots, and ensemble/best-model monitoring when enabled.

### Data Format

Training and test data are stored as MATLAB `.mat` files.

For SDE and SPDE examples, the usual variable is:

```text
data
```

with shape:

```text
[dim, number_of_time_steps, number_of_trajectories]
```

Nonautonomous examples may additionally use a parameter or excitation array, often named:

```text
para
```

with shape similar to:

```text
[parameter_dimension, number_of_time_steps, number_of_trajectories]
```

SSA examples may use trajectory data, conditional-distribution data, or original path data depending on the production script. Conditional SSA files may use dictionary-style keys such as:

```text
0_i, 0_d, 1_i, 1_d, ...
```

where `*_i` stores the fixed initial condition and `*_d` stores samples from the corresponding conditional distribution.

## Model Execution

### Autonomous SDE and Markovian Effective Dynamics of Multiscale SDEs

This category uses NF and ResNF models for autonomous stochastic dynamics and Markovian effective dynamics of multiscale SDEs.

NF runs use:

```bash
python SolveNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=NFSDE
```

Residual NF runs use:

```bash
python SolveResNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=ResNFSDE
```


### Nonautonomous SDEs

This category uses NF and ResNF models for systems with time-dependent forcing, controls, or external excitation.

NF nonautonomous runs use:

```bash
python SolveNFNonAutoSDE.py --test_name=<run_name> --config_path=<config_path>
```

Residual nonautonomous runs use:

```bash
python SolveResNFNonAutoSDE.py --test_name=<run_name> --config_path=<config_path>
```


### SSA / Chemical Reaction Networks

This category uses NF, ResNF, and MixNF models for SSA examples.

NF or conservative NF SSA runs use:

```bash
python SolveNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=NFSDE
```

or, for conservative SSA variants:

```bash
python SolveNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=NFSDE_SSAconserve
```

Residual SSA runs use:

```bash
python SolveResNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=ResNFSDE
```

or, for residual/conservative SSA variants:

```bash
python SolveResNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=ResNFSDE_SSAgenconserve
```

MixNF runs use an ODE or chemical-dynamics prior and learn the remaining stochastic component around that prior:

```bash
python SolveMixNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=MixNFSDE
```

or, for mixed/conservative SSA variants:

```bash
python SolveMixNFSDE.py --test_name=<run_name> --config_path=<config_path> --model_name=MixNFSDE_SSAgenconserve
```


### Post Test and Validation

To reload a trained model and generate predictions:

```bash
python ShowTest.py --test_name=<example> --model_name=<model_name>
```

To generate production or paper-style plots:

```bash
python ShowProdcution.py --test_name=<example> --model_name=<model_name> --test_case=<case_name>
```

Additional performance plots can be generated with:

```bash
python ShowPerformance.py --test_name=<example>
```


## Model Outputs

Training writes outputs under:

```text
results/<test_name>/
```

A full run may generate:

```text
results/<test_name>/Test_config.json
results/<test_name>/Test_model/
results/<test_name>/Test_history.json
results/<test_name>/predict.mat
results/<test_name>/Monitor/
```

`Test_config.json` records the run settings. `Test_model/` stores the final trained model weights. `Test_history.json` records training history when written. `predict.mat` stores generated prediction trajectories. `Monitor/` stores monitoring outputs, checkpoints, and diagnostics.

The cleaned result folders in this repository intentionally keep only the files needed for reproduction:

```text
results/<example>/Test_config.json
results/<example>/Test_model/
```

Some examples also keep additional trained checkpoints:

```text
results/<example>/Monitor/Ens_model/
results/<example>/Monitor/Best_model/
```

`Monitor/Ens_model/` stores ensemble checkpoints when ensemble prediction was used. `Monitor/Best_model/` stores the best checkpoint when best-model monitoring was used. Generated plots, logs, `predict.mat`, metrics folders, and production figure folders are not included in the cleaned result package by default.
