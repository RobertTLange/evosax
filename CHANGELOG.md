### Work-in-Progress

- Implement more strategies
    - [ ] Large-scale CMA-ES variants
        - [ ] [LM-CMA](https://www.researchgate.net/publication/282612269_LM-CMA_An_alternative_to_L-BFGS_for_large-scale_black_Box_optimization)
        - [ ] [VkD-CMA](https://hal.inria.fr/hal-01306551v1/document), [Code](https://gist.github.com/youheiakimoto/2fb26c0ace43c22b8f19c7796e69e108)
    - [ ] [ASEBO](https://proceedings.neurips.cc/paper/2019/file/88bade49e98db8790df275fcebb37a13-Paper.pdf)
    - [ ] [RBO](http://proceedings.mlr.press/v100/choromanski20a/choromanski20a.pdf)

- Encoding methods - via special reshape wrappers
    - [ ] Discrete Cosine Transform
    - [ ] Wavelet Based Encoding (van Steenkiste, 2016)
    - [ ] CNN Hypernetwork (Ha - start with simple MLP)

### [v0.1.0] - [TBD]

##### Added

- Adds a `total_env_steps` counter to both `GymFitness` and `BraxFitness` for easier sample efficiency comparability with RL algorithms.
- Support for new strategies/genetic algorithms
    - SAMR-GA (Clune et al., 2008)
    - GESMR-GA (Kumar et al., 2022)
    - SNES (Wierstra et al., 2014)
    - DES (Lange et al., 2022)
    - Guided ES (Maheswaranathan et al., 2018)
    - ASEBO (Choromanski et al., 2019)
- Adds full set of BBOB low-dimensional functions (`BBOBFitness`)
- Adds 2D visualizer animating sampled points (`BBOBVisualizer`)
- Adds `Evosax2JAXWrapper` to wrap all evosax strategies
- Adds Adan optimizer (Xie et al., 2022)

##### Changed

- `ParameterReshaper` can now be directly applied from within the strategy. You simply have to provide a `pholder_params` pytree at strategy instantiation (and no `num_dims`).
- `FitnessShaper` can also be directly applied from within the strategy. This makes it easier to track the best performing member across generations and addresses issue #32. Simply provide the fitness shaping settings as args to the strategy (`maximize`, `centered_rank`, ...)
- Removes Brax fitness (use EvoJAX version instead)

##### Fixed

- Fixed reward masking in `GymFitness`. Using `jnp.sum(dones) >= 1` for cumulative return computation zeros out the final timestep, which is wrong. That's why there were problems with sparse reward gym environments (e.g. Mountain Car).
- Fixed PGPE sample indexing.

### [v0.0.9] - 15/06/2022

##### Added

- Base indirect encoding methods in `experimental`. Sofar support for:
    - Random projection-based decodings
    - Hypernetworks for MLP architectures
- Example notebook for infirect encodings.
- Example notebook for Brax control tasks and policy visualizations.
- Adds option to restart wrappers to `copy_mean` and only reset other parts of `EvoState`.

##### Changed

- Change problem wrappers to work with `{"params": ...}` dictionary. No longer need to define `ParameterReshaper(net_params["params"])` to work without preselecting "params". Changed tests and notebooks accordingly.
- Restructured all strategies to work with flax structured dataclass and `EvoState`/`EvoParams`.

```python
from flax import struct
@struct.dataclass
class State:
    ...
```

- The core strategy API now also works without `es_params` being supplied in call. In this case we simply use the default settings.
- Moved all gym environment to (still private but soon to be released) `gymnax`.
- Updated all notebooks accordingly.

##### Fixed

- Makes `ParameterReshaper` work also with `dm-haiku`-style parameter dictionaries. Thanks to @vuoristo.

### [v0.0.8] - [24/05/2022]

##### Fixed

- Fix gym import bug and codecov patch tolerance.

### [v0.0.7] - [24/05/2022]

##### Fixed

- Bug due to `subpops` import in `experimental`.

### [v0.0.6] - [24/05/2022]

##### Added

- Adds basic indirect encoding method in `experimental` - via special reshape wrappers: `RandomDecoder` embeddings (Gaussian, Rademacher)

##### Fixed

- Fix import of modified ant environment. Broke due to optional brax dependence.
##### Changed

- Restructure batch strategies into `experimental`
- Make ant modified more flexible with configuration dict option (`modify_dict`)

### [v0.0.5] - [22/05/2022]

##### Added

- Adds sequential problems (SeqMNIST and MNIST) to evaluation wrappers.
- Adds Acrobot task to `GymFitness` rollout wrappers.
- Adds modified Ant environment to Brax rollout.
- New strategies:
    - RmES (`RmES` following Li & Zhang, 2008).
    - Gradientless Descent (`GLD` following Golovin et al., 2020).
    - Simulated Annealing (`SimAnneal` following Rasdi Rere et al., 2015)
- Adds simultaneous batch strategy functionalities:
    - `BatchStrategy`: `vmap`/`pmap` distributed subpopulation rollout
    - `Protocol`: Communication protocol between subpopulations
    - `MetaStrategy`: Stack one ES on top of subpopulations to control hyperparameters

##### Changed

- Renamed `crossover_rate` to `cross_over_rate` in DE to make consistent with `SimpleGA`.
- Add option to add optional `env_params` to `GymFitness`, `seq_length` to addition and `permute_seq` for S-MNIST problem.
- Network classes now support different initializers for the kernels using the `kernel_init_type` string option. By default we follow flax's choice in `lecun_normal`.

##### Fixed

- Add `spring_legacy` option to Brax rollout wrappers.

### [v0.0.4] - [26/03/2022]

##### Added

- New strategies:
    - Separable CMA-ES strategy (`Sep_CMA_ES` following Ros & Hansen, 2008).
    - BIPOP-CMA-ES (`BIPOP_CMA_ES`, following Hansen, 2009)
    - IPOP-CMA-ES (`IPOP_CMA_ES`, following Auer & Hansen, 2005)
    - Full-iAMaLGaM (`Full_iAMaLGaM`, following Bosman et al., 2013)
    - MA-ES (`MA_ES`, following Bayer & Sendhoff, 2017)
    - LM-MA-ES (`LM_MA_ES`, following Loshchilov et al., 2017)
- Restart wrappers: 
    - Base restart class (`RestartWrapper`).
    - Simple reinit restart strategy (`Simple_Restarter`).
    - BIPOP strategy with interleaved small/large populations (`BIPOP_Restarter`).
    - IPOP strategy with growing population size (`IPOP_Restarter`).

##### Changed

- Both `ParamReshaper` and the rollout wrappers now support `pmap` over the population member dimension.
- Add `mean` state component to all strategies (also non-GD-based) for smoother evaluation protocols.
- Add `strategy_name` to all strategies.
- Major renaming of strategies to more parsimonious version (e.g. `PSO_ES` -> `PSO`)

##### Fixed

- Fix `BraxFitness` rollout wrapper and add train/test option.
- Fix small bug related to sigma decay in `Simple_GA`.
- Increase numerical stability constant to 1e-05 for z-scoring fitness reshaping. Everything smaller did not work robustly.
- Get rid of deprecated `index_update` and use `at[].set()`

### [v0.0.3] - 21/02/2022

- Fix Python version requirements for evojax integration. Needs to be >3.6 since JAX versions stopped supporting 3.6.

### [v0.0.2] - 17/02/2022

- First public release including 11 strategies, rollout/network wrappers, utilities.

### [v0.0.1] - 22/11/2021

##### Added
- Adds all base functionalities
