# Stochastic Flow Map Learning with Normalizing Flows

This repository contains normalizing-flow implementations for stochastic flow map learning. The code supports three related model forms:

- **NF**: a direct normalizing-flow model for stochastic flow-map transitions, $x_{n+1} = G_\theta(x_n, z)$.
- **ResNF**: a residual normalizing-flow model, $x_{n+1} = x_n + G_\theta(x_n, z)$.
- **MixNF**: a mixed normalizing-flow model, $x_{n+1} = \Phi(x_n) + G_\theta(x_n, z)$. In the SSA examples from the second SSA paper, $\Phi$ is an ODE or chemical-dynamics prior, and the normalizing flow learns the remaining stochastic component around that prior.

The examples in this repository apply these models to three settings:

- ordinary stochastic differential equations, including autonomous and nonautonomous systems;
- Markovian effective dynamics of multiscale SDEs;
- stochastic simulation algorithm models for chemical reaction networks.

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

### Nonautonomous and Controlled SDEs

These examples correspond to [1].

| Example | Result folder | Description | Model type |
| --- | --- | --- | --- |
| Ex12 | `results/Ex12` | OU process with drift control | NF |
| Ex15 | `results/Ex15` | Stochastic resonance / double-well with excitation | NF |
| Ex16 | `results/Ex16` | Nonlinear SDE with control | NF |
| Ex17 | `results/Ex17` | Stochastic predator-prey model with excitation | NF |
| Ex19 | `results/Ex19` | OU process with both drift and diffusion control | NF |
| Ex43 | `results/Ex43` | Gene expression SSA model with time-dependent reaction rate | ResNF |
| SPDEEx3 | `results/SPDEEx3` | Stochastic heat equation with source, modal/spectral form | ResNF |

### Markovian Effective Dynamics of Multiscale SDEs

These examples correspond to [2].

| Example | Result folder | Description | Model type |
| --- | --- | --- | --- |
| Ex28 | `results/Ex28` | Skew product SDE | NF |
| Ex33 | `results/Ex33` | Exponential mean OU / multiscale exponential example | NF |
| Ex38 | `results/Ex38` | Triad system | NF |
| Ex34 | `results/Ex34` | 3D nonlinear multiscale SDE | NF |
| Ex36 | `results/Ex36` | Multiscale stochastic oscillator | NF |

### SSA / Chemical Reaction Networks

These examples correspond to [3] and [4].

| Paper | Example | Result folder | Description | Model type |
| --- | --- | --- | --- | --- |
| [3] | Ex22 | `results/Ex22` | Transfer process | Conservative NF |
| [3] | Ex23 LV slow | `results/Ex23_LVSlow` | Slow Lotka-Volterra SSA model | NF |
| [3] | Ex23 LV fast | `results/Ex23_LVFast` | Fast Lotka-Volterra SSA model | ResNF |
| [3] | Ex25 | `results/Ex25` | Brusselator | NF |
| [3] | Ex27 | `results/Ex27` | Autocatalysis | Conservative ResNF |
| [3] | Ex26 | `results/Ex26` | Oregonator | ResNF |
| [4] | Ex45 | `results/Ex45` | Schlogl model | MixNF |
| [4] | Ex42 | `results/Ex42` | Vilar 2002 genetic oscillator model | Conservative MixNF |
| [4] | Ex41 | `results/Ex41` | Mammalian circadian clock model | MixNF |
| [4] | Ex23 Mix | `results/Ex23_Mix` | Lotka-Volterra SSA model with mixed prior | MixNF |

## Preparation

### Config

Each run is controlled by a config file. For cleaned result folders, `results/<example>/Test_config.json` records the settings used for that trained model. Not every example uses every parameter below; example-specific parameters are read only by the corresponding equation, model, or plotting routine.

`eqn_config`: equation, prior, and physical-model settings.

- `_comment`: human-readable note for the example.
- `eqn_name`: equation, SDE, SPDE, or SSA example name used by the evaluation and plotting code.
- `dim`: state dimension.
- `dim_para`: dimension of the external parameter, control, or forcing input for nonautonomous examples.
- `Delta`: time-step size used by the flow map.
- `para`: fixed parameter vector for examples that store equation parameters directly in the config.
- `mu`, `sigma`, `theta`: standard drift/diffusion parameters used by OU-type or related SDE examples.
- `sigma_`, `sigma_1`, `sigma_2`, `sigma_3`: component-wise noise strengths used by multicomponent examples.
- `alpha`, `gamma`, `lambda_`, `epsilon`: example-specific coefficients used by multiscale, oscillator, or reaction-network examples.
- `p`, `q`, `V`, `f_`: example-specific physical or reaction parameters.
- `omega`, `omega2`: frequency parameters for periodically forced or oscillatory examples.
- `s1`, `s2`: noise/control scale parameters used by selected nonautonomous examples.
- `resmodel`: deterministic/prior model used by ResNF or MixNF examples. Examples include `Exact` and `ChemicalODE`.
- `resmodel_path`: path to the pretrained deterministic/prior model checkpoint when the run uses one.
- `resconfig_path`: path to the config file for the deterministic/prior model when the run uses one.

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
- `weight_decay`: weight-decay coefficient used by the optimizer.
- `Test_mode`: prediction/testing mode used during or after training. The included configs commonly use `Normal`.
- `Note`: optional human-readable training note.
- `l_rate`: legacy scalar learning-rate field.
- `l_rate_config`: active learning-rate configuration selected for the run.
- `l_rate_config_value`: fixed learning-rate option.
  - `name`: scheduler name, usually `value`.
- `l_rate_config_step`: step-decay learning-rate option.
  - `name`: scheduler name.
  - `step`: decay step interval.
  - `gamma`: multiplicative decay factor.
- `l_rate_config_cyclic`: cyclic learning-rate option.
  - `name`: scheduler name.
  - `base`: lower learning-rate bound.
  - `max`: upper learning-rate bound.
  - `step`: cycle step length.
  - `gamma`: decay factor for cycle amplitude when used.
- `l_rate_config_Stepcyclic`: step-cyclic learning-rate option.
  - `name`: scheduler name.
  - `base`, `max`, `step`, `gamma`: cyclic scheduler parameters.
  - `scale`: additional scaling factor.
  - `gstep`: global step interval for scaling.
- `l_rate_config_ROnPlat`: reduce-on-plateau learning-rate option.
  - `name`: scheduler name.
  - `minr`: minimum learning rate.
  - `factor`: reduction factor.
  - `patience`: patience before reducing the learning rate.

`dat_config`: data paths, prediction size, and sampling settings.

- `TrainData_dir`: path to the training `.mat` file.
- `TestData_dir`: path to the test `.mat` file.
- `n_ea_traj`: number of sampled training windows or trajectory segments per trajectory.
- `N_pred`: number of prediction trajectories used by the standard prediction routines.
- `Test_mode`: testing data mode. The included configs commonly use `Normal`.
- `pair_data`: whether the training data are stored as paired input/output transitions.
- `DiscretePred`: whether prediction is treated as a discrete-step prediction problem.
- `ConstrainedPred`: whether prediction uses conservation or feasibility constraints.
- `N_train_base`: base number of training samples used by selected runner-generated configs.

`show_config`: standard postprocessing switches.

- `plot_samplecompare`: whether to generate sample trajectory comparison plots.
- `plot_meancompare`: whether to generate mean/std comparison plots.
- `plot_losthist`: whether to generate loss-history plots.

`monitor_config`: training-time and validation-time diagnostics.

- `traindata_hist`: whether to plot training-data histograms.
- `traintransin_hist`: whether to plot transition-input histograms.
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
- `Ens_monitor`: ensemble-checkpoint monitor.
  - `if`: enable or disable ensemble checkpointing.
  - `Ens_repdf`: ensemble density/PDF diagnostics.
  - `Ens_cond_mv`: ensemble conditional mean/variance diagnostics.
  - `Ens_eva`: ensemble mean/std evaluation diagnostics.
- `Best_monitor`: best-model checkpoint monitor.
  - `if`: enable or disable best-model monitoring.
  - `Best_repdf`: density/PDF diagnostics for the best model.
  - `Best_cond_mv`: conditional mean/variance diagnostics for the best model.
  - `Best_eva`: mean/std evaluation diagnostics for the best model.
  - `sample`: optional sample count for selected examples.
- `stoppingtime`: stopping-time diagnostics for selected SSA examples.
  - `if`: enable or disable stopping-time evaluation.
  - `path`: output or reference path.

Other top-level metadata fields may appear in runner-generated configs:

- `runner_note`: note added by a training or rerun script.
- `run_metadata`: bookkeeping information such as `base_config`, `train_data`, `test_data`, `boundary_train_data`, `n_ea_traj`, and a runner note. These fields document how a rerun was constructed and are not usually model hyperparameters.

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
