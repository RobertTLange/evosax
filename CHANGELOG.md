### Work-in-Progress

- [ ] Make `weights` and `weights_truncated` part of CMA-ES state due to shape dependence
    - [ ] Clean up BIPOP/IPOP ask afterwards
- [ ] Add brax example notebook
- [ ] Add batch strategy wrapper
- [ ] Change network wrapper to work with list of hidden neurons?
- [ ] How can we make restart wrappers be jittable (problem of non-static population sizes)?
- More strategies
    - [ ] Large-scale CMA-ES variants
        - [ ] LM-CMA
        - [ ] VkD-CMA
        - [ ] RmES
    - [ ] sNES (separable)

### [v0.0.4] - [26/03/2022]

##### Added

- Add `spring_legacy` option to Brax rollout wrappers

##### Changed

##### Fixed

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
