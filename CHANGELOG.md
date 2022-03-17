### TODO

- [ ] Change fitness reshaping to be part of strategy - makes storage of best fitness better (no trafo stored)?
- [ ] Change network wrapper to work with list of hidden neurons?
- [ ] Update notebooks for new rollout wrappers
- [ ] Add restart wrapper notebook
- [ ] Add batch strategy wrapper
- [ ] Refine default hyperparameters based on gridsearches
- [ ] How can we make restart wrappers be jittable (problem of non-static population sizes)?
- More strategies
    - [ ] iAmalgam-IDEA strategies
        - [ ] independent
        - [ ] full
    - [ ] Large-scale CMA-ES variants
        - [ ] Cholesky CMA-ES
        - [ ] LMA-CMA
        - [ ] VkD-CMA
        - [ ] RmES
    - [ ] sNES (separable)


### [v0.0.4] - [Unreleased]

##### Added

- New strategies:
    - Separable CMA-ES strategy (`Sep_CMA_ES` following Ros & Hansen, 2008).
    - BIPOP-CMA-ES (`BIPOP_CMA_ES`, following Hansen, 2009)
    - IPOP-CMA-ES (`IPOP_CMA_ES`, following Auer & Hansen, 2005)
- Restart wrappers: 
    - Base restart class (`RestartWrapper`).
    - Simple reinit restart strategy (`Simple_Restarter`).
    - BIPOP strategy with interleaved small/large populations (`BIPOP_Restarter`).
    - IPOP strategy with growing population size (`IPOP_Restarter`).

##### Changed

- Both `ParamReshaper` and the rollout wrappers now support `pmap` over the population member dimension.
- Add `mean` state component to all strategies (also non-GD-based) for smoother evaluation protocols.
- Add `strategy_name` to all strategies.

##### Fixed

- Fix `BraxFitness` rollout wrapper and add train/test option.
- Fix small bug related to sigma decay in `Simple_GA`.
- Increase numerical stability constant to 1e-05 for z-scoring fitness reshaping. Everything smaller did not work robustly. 

### [v0.0.3] - 21/02/2021

- Fix Python version requirements for evojax integration. Needs to be >3.6 since JAX versions stopped supporting 3.6.

### [v0.0.2] - 17/02/2021

- First public release including 11 strategies, rollout/network wrappers, utilities.

### [v0.0.1] - 22/11/2021

##### Added
- Adds all base functionalities
