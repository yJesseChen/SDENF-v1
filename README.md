# Stochastic Flow Map Learning with Normalizing Flows

This repository contains normalizing-flow implementations for stochastic flow map learning. The code supports three related model forms:

- **NF**: a direct normalizing-flow model for stochastic flow-map transitions, $x_{n+1} = G_\theta(x_n, z)$.
- **ResNF**: a residual normalizing-flow model, $x_{n+1} = x_n + G_\theta(x_n, z)$.
- **MixNF**: a mixed normalizing-flow model, $x_{n+1} = \Phi(x_n) + G_\theta(x_n, z)$. In the SSA examples, $\Phi$ is an ODE or chemical-dynamics prior, and the normalizing flow learns the remaining stochastic component around that prior.

The examples in this repository apply these models to three settings:

- ordinary stochastic differential equations, including autonomous and nonautonomous systems;
- Markovian effective dynamics of multiscale SDEs;
- stochastic simulation algorithm models for chemical reaction networks.

A step-by-step tutorial for the Ex4 exponential-diffusion example is provided in [`tutorials/Ex4_tutorial.ipynb`](tutorials/Ex4_tutorial.ipynb). The tutorial walks through the task setup, data loading and plotting, config inspection, training, model outputs, prediction, and postprocessing.

## References

The included examples are associated with the following papers:

- [1] Yuan Chen and Dongbin Xiu, `Modeling Unknown Stochastic Dynamical System Subject to External Excitation`, 2026.
  https://iamyuanchen.xyz/pdf/2026ChenXiu.pdf
- [2] Yuan Chen and Dongbin Xiu, `Data-Driven Effective Modeling of Multiscale Stochastic Dynamical Systems`, 2024.
  https://iamyuanchen.xyz/pdf/2024ChenXiu_b.pdf
- [3] Yuan Chen, Weize Mao, and Dongbin Xiu, `Data-Driven Effective Modeling of Stochastic Chemical Reaction Networks`, to be published soon.

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

A detailed record of the local `DeepLearningTest` conda environment is included in [`environment-deeplearningtest.yml`](environment-deeplearningtest.yml). To recreate a similar environment, use:

```bash
conda env create -f environment-deeplearningtest.yml
```

## Repository Structure

The main folders are:

| Folder | Contents |
| --- | --- |
| `config/` | Example JSON configs. The current cleaned copy keeps a small representative config for the autonomous SDE example. |
| `data/` | Local/generated MATLAB `.mat` datasets. For the Ex4 tutorial, generate data with `SDEDATA-v1/Ex4ExpDiff.py` and place the files under `data/Ex4ExpDiff/`. Data files are not included by default. |
| `results/` | Cleaned result folders for the included paper examples. Each folder keeps config and trained weights only. |
| `src/` | Supporting source files used by the flow models. |

The main files are grouped by model type in the table below:

| File | Purpose |
| --- | --- |
| **NF** |  |
| `SolveNFSDE.py` | Training entry point for ordinary NF SDE models, including autonomous SDE examples and some SSA examples. |
| `NFSDE.py` | Core NF SDE model implementation. |
| `SolveNFNonAutoSDE.py` | Training entry point for nonautonomous NF SDE models. |
| `NFNonAutoSDE.py` | Nonautonomous NF model implementation. |
| `NFSDE_SSAconserve.py` | Conservative NF variant for SSA examples. |
| **ResNF** |  |
| `SolveResNFSDE.py` | Training entry point for residual NF SDE models. |
| `ResNFSDE.py` | Core residual NF SDE model implementation. |
| `SolveResNFNonAutoSDE.py` | Training entry point for residual nonautonomous NF SDE models. |
| `ResNFNonAutoSDE.py` | Nonautonomous residual NF model implementation. |
| `ResNFSDE_SSAgenconserve.py` | Residual/conservative NF variant for SSA examples. |
| **MixNF** |  |
| `SolveMixNFSDE.py` | Training entry point for mixed NF models. |
| `MixNFSDE.py` | Core mixed NF SDE model implementation. |
| `SolveMixNFNonAutoSDE.py` | Training entry point for mixed nonautonomous NF SDE models. |
| `MixNFNonAutoSDE.py` | Nonautonomous mixed NF model implementation. |
| `MixNFSDE_SSAgenconserve.py` | Mixed/conservative NF variant for SSA examples. |
| `Chemical_Dynamics.py` | Chemical-dynamics helper functions used by MixNF SSA examples. |
| **Post Validation and Plots** |  |
| `ShowTest.py` | Testing and prediction entry point for trained models. |
| `ShowProdcution.py` | Postprocessing entry point for production/paper-style figures. |
| `ShowPerformance.py` | Additional performance and diagnostic plotting utility. |
| `Evaulation.py` and `Prodcution.py` | Evaluation and production plotting utilities. |

## Included Example

The cleaned `results/` folder contains trained configs and weights for the following paper examples. We only provide configuration files and trained weights needed for reproduction; generated plots, logs, `predict.mat`, metrics folders, and production figure folders are not included by default.

**Generated `.mat` data files are not included in this repository by default. Generate the required data with `SDEDATA-v1`, or place your own data files under the paths specified by `dat_config` in the corresponding config.**

The `--model_name` column gives the model selector to pass to `ShowTest.py`, `ShowProdcution.py`, and the corresponding training script when reproducing that example.

### Nonautonomous and Controlled SDEs

These examples correspond to [1].

| Example | Result folder | Description | `--model_name` |
| --- | --- | --- | --- |
| Ex12 | `results/Ex12` | OU process with drift control | `NFNonAutoSDE` |
| Ex15 | `results/Ex15` | Stochastic resonance / double-well with excitation | `NFNonAutoSDE` |
| Ex16 | `results/Ex16` | Nonlinear SDE with control | `NFNonAutoSDE` |
| Ex17 | `results/Ex17` | Stochastic predator-prey model with excitation | `NFNonAutoSDE` |
| Ex19 | `results/Ex19` | OU process with both drift and diffusion control | `NFNonAutoSDE` |
| Ex43 | `results/Ex43` | Gene expression SSA model with time-dependent reaction rate | `ResNFNonAutoSDE` |
| SPDEEx3 | `results/SPDEEx3` | Stochastic heat equation with source, modal/spectral form | `ResNFNonAutoSDE` |

### Markovian Effective Dynamics of Multiscale SDEs

These examples correspond to [2].

| Example | Result folder | Description | `--model_name` |
| --- | --- | --- | --- |
| Ex28 | `results/Ex28` | Skew product SDE | `NFSDE` |
| Ex33 | `results/Ex33` | Exponential mean OU / multiscale exponential example | `NFSDE` |
| Ex38 | `results/Ex38` | Triad system | `NFSDE` |
| Ex34 | `results/Ex34` | 3D nonlinear multiscale SDE | `NFSDE` |
| Ex36 | `results/Ex36` | Multiscale stochastic oscillator | `NFSDE` |

### SSA / Chemical Reaction Networks

These examples correspond to [3].

| Example | Result folder | Description | `--model_name` |
| --- | --- | --- | --- |
| Ex22 | `results/Ex22` | Transfer process | `NFSDE_SSAconserve` |
| Ex23&nbsp;LV&nbsp;slow | `results/Ex23_LVSlow` | Slow Lotka-Volterra SSA model | `NFSDE` |
| Ex23&nbsp;LV&nbsp;fast | `results/Ex23_LVFast` | Fast Lotka-Volterra SSA model | `ResNFSDE` |
| Ex25 | `results/Ex25` | Brusselator | `NFSDE` |
| Ex27 | `results/Ex27` | Autocatalysis | `ResNFSDE_SSAgenconserve` |
| Ex26 | `results/Ex26` | Oregonator | `ResNFSDE` |
| Ex45 | `results/Ex45` | Schlogl model | `MixNFSDE` |
| Ex42 | `results/Ex42` | Vilar 2002 genetic oscillator model | `MixNFSDE_SSAgenconserve` |
| Ex41 | `results/Ex41` | Mammalian circadian clock model | `MixNFSDE` |
| Ex23&nbsp;Mix | `results/Ex23_Mix` | Lotka-Volterra SSA model with mixed prior | `MixNFSDE` |

## Preparation

### Config

Each run is controlled by a config file. For cleaned result folders, `results/<example>/Test_config.json` records the settings used for that trained model. Not every example uses every parameter below; example-specific parameters are read only by the corresponding equation, model, or plotting routine.

<p><strong><span style="color:#0969da">For a first run, I recommend starting from the config of a similar provided example and replacing the data paths with your own data. The `monitor_config` section is used to monitor performance during training; it is useful but not mandatory. <u>You can set all monitor `if` fields to `false` to avoid complex errors.</u></span></strong></p>

Here is the full list of those parameters:

`eqn_config`: equation and physical-model settings.

- `_comment`: human-readable note for the example.
- `eqn_name`: equation, SDE, SPDE, or SSA example name used by the evaluation and plotting code.
- `dim`: state dimension.
- `dim_para`: dimension of the external parameter, control, or forcing input for nonautonomous examples.
- `Delta`: time-step size used by the flow map.
- Model parameters: example-specific equation, physical parameters.
- Prior model: only used by Mix models.
  - `resmodel`: deterministic/prior model. Examples include `Exact` and `ChemicalODE`.
  - `resmodel_path`: path to the pretrained deterministic/prior model checkpoint.
  - `resconfig_path`: path to the config file for the deterministic/prior model.

`net_config`: normalizing-flow architecture and training settings.

- `fname`: flow architecture name. The provided examples mainly use masked autoregressive flow, `MAF`.
- `flevel`: number of flow blocks or flow levels.
- `net_spec`: neural-network specification for the flow transform.
  - `nodes`: number of nodes per hidden layer.
  - `layer`: number of hidden layers.
  - `act`: activation function.
- `N_rec`: number of consecutive transition steps used by the model during training or prediction. In the included configs this is typically `2`.
- `batch_size`: training batch size.
- `N_epochs`: number of training epochs.
- `weight_decay`: weight-decay coefficient used by the optimizer for regularization and to help mitigate overfitting.
- `Test_mode`: prediction/testing mode used during or after training. The included configs commonly use `Normal`.
- `Note`: optional human-readable training note.
- `l_rate`: legacy scalar learning-rate field.
- `l_rate_config`: active learning-rate configuration selected for the run. It can be one of the following options:
  - `l_rate_config_value`: fixed learning rate.
  - `l_rate_config_step`: step-decay learning rate, controlled by `step` and `gamma`.
  - `l_rate_config_cyclic`: cyclic learning rate, controlled by `base`, `max`, `step`, and `gamma`.
  - `l_rate_config_Stepcyclic`: step-cyclic learning rate, controlled by `base`, `max`, `step`, `gamma`, `scale`, and `gstep`.
  - `l_rate_config_ROnPlat`: reduce-on-plateau learning rate, controlled by `minr`, `factor`, and `patience`.

`dat_config`: data paths, prediction size, and sampling settings.

- `TrainData_dir`: path to the training `.mat` file.
- `TestData_dir`: path to the test `.mat` file.
- Training sample construction: controlled by `pair_data`, `n_ea_traj`, and `N_train_base`.
  - If `pair_data` is absent or `false`, the code treats `data` as long trajectories with shape `[dim, number_of_time_steps, number_of_trajectories]`. It randomly samples `n_ea_traj` short windows from each long trajectory, so the effective training size is `number_of_trajectories * n_ea_traj`.
  - If `pair_data` is `true`, the code treats `data` as precomputed input/output transition pairs. For the usual `N_rec = 2` case, the expected shape is `[dim, 2, number_of_pairs]`, where `data[:, 0, :]` stores inputs and `data[:, 1, :]` stores outputs.
    - In pair-data mode, `N_train_base` is the number of base transition pairs used for each repeat, and the effective training size is `N_train_base * n_ea_traj`.
    - For example, `N_train_base = 10000` and `n_ea_traj = 12` gives `120000` training pairs. The pair-data file should contain at least `120000` available pairs.
- `N_pred`: number of prediction trajectories used by standard prediction and monitoring routines. In practice, set this to match the number of trajectories in `TestData_dir` for the usual test workflow; ensemble and monitoring code use it when allocating prediction arrays.
- `Test_mode`: testing data mode. The included configs commonly use `Normal`.
- `DiscretePred`: whether prediction output should be projected to discrete count data. If enabled, after each model prediction the code sets negative values to `0` and rounds the result to integers. This is mainly useful for SSA/count-valued examples.
- `ConstrainedPred`: whether prediction should enforce a simple feasibility constraint. In the current code this is used for `SSALV`; invalid trajectories that hit near-zero population values are filtered out during recursive prediction, and the code repeats prediction attempts until it fills the requested test-data size or reaches the retry limit.

`show_config`: standard postprocessing switches.

- `plot_samplecompare`: whether to generate sample trajectory comparison plots.
- `plot_meancompare`: whether to generate mean/std comparison plots.
- `plot_losthist`: whether to generate loss-history plots.

`monitor_config`: training-time and validation-time diagnostics.

- `traindata_hist`: whether to plot a two-dimensional hit diagram of the sampled training trajectories. The plot shows time on the horizontal axis, state value on the vertical axis, and point density by color; files are saved under `Monitor/dataplot/hist_Train_data*.png`.
- `traintransin_hist`: whether to plot histograms of transition inputs `X_s` used for training. These plots help check whether the sampled transition inputs cover the expected state range; files are saved under `Monitor/dataplot/hist_input_Train_data*.png`.
- `repdf_display`: repeated density/PDF display settings.
  - `if`: enable or disable the monitor.
  - `size`: number of samples used for the display when present.
  - `range`: plotting range.
  - `times`: training epochs or iterations at which to plot.
  - `int_long`: integration/prediction length for the display when present.
  - `path`: output or reference path for selected SSA displays.
- `cond_mv`: conditional mean/variance monitor.
  - `if`: enable or disable the monitor.
  - `Npoint`: number of conditioning points.
  - `range`: conditioning range.
  - `times`: epochs or iterations at which to evaluate.
- `Evameanv`: recursive prediction and mean/std evaluation monitor.
  - `if`: enable or disable the monitor.
  - `times`: epochs or iterations at which to evaluate.
  - `type`: evaluation type, usually `Normal`.
  - `sample`: optional sample count for selected examples.
- `loss`: loss plotting monitor.
  - `if`: enable or disable the monitor.
  - `times`: epochs or iterations at which to save the loss plot.
- `Ens_monitor`: ensemble-checkpoint monitor. When enabled, the code saves multiple model checkpoints at scheduled late-stage epochs under `Monitor/Ens_model/`. At selected endpoint epochs, it reloads those saved models and can generate ensemble diagnostics.
  - `if`: enable or disable ensemble checkpointing.
  - `Ens_repdf`: generate ensemble repeated density/PDF diagnostics.
  - `Ens_cond_mv`: generate ensemble conditional mean/variance diagnostics.
  - `Ens_eva`: generate ensemble recursive prediction mean/std diagnostics.
- `Best_monitor`: best-model checkpoint monitor. When enabled, the code tracks the best monitored objective during training and saves the current best model under `Monitor/Best_model/model.pt`. At scheduled evaluation times, it can reload that best model and generate diagnostics.
  - `if`: enable or disable best-model monitoring.
  - `Best_repdf`: generate density/PDF diagnostics using the best model.
  - `Best_cond_mv`: generate conditional mean/variance diagnostics using the best model when supported.
  - `Best_eva`: generate recursive prediction mean/std diagnostics using the best model.
  - `sample`: optional sample-count setting for selected examples.
- `stoppingtime`: stopping-time diagnostics for selected SSA examples.
  - `if`: enable or disable stopping-time evaluation.
  - `path`: output or reference path.

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

Common command-line arguments:

- `test_name`: name of the run or saved result folder. Training writes outputs to `results/<test_name>/`; testing and plotting reload from this folder.
- `config_path`: path to the JSON config file used for training.
- `model_name`: model class selected by the script, such as `NFSDE`, `ResNFSDE`, `MixNFSDE`, `NFNonAutoSDE`, `ResNFNonAutoSDE`, `NFSDE_SSAconserve`, `ResNFSDE_SSAgenconserve`, or `MixNFSDE_SSAgenconserve`.
- `test_case`: postprocessing case name used by `ShowProdcution.py`. Different values correspond to different test datasets, prediction horizons, or figure-production settings for the same trained model.

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

NF SSA runs use:

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

## Minimal Reproduction Workflow

For a saved example in `results/<example>/`, the usual reproduction workflow is:

1. Generate or copy the required `.mat` data files into the paths specified by `results/<example>/Test_config.json`.
2. Run prediction with the model selector shown in the `--model_name` column:

```bash
python ShowTest.py --test_name=<example> --model_name=<model_name> --test_case=<case_name>
```

3. Generate postprocessing or paper-style figures from the prediction output:

```bash
python ShowProdcution.py --test_name=<example> --model_name=<model_name> --test_case=<case_name>
```

Prediction and figure outputs are written under `results/<example>/<case_name>/` and are not tracked by git.

