### TODO

- [ ] Change network wrapper to work with list of hidden neurons?
- [ ] Update notebooks for new rollout wrappers
- [ ] Add restart wrapper notebook
- [ ] Update selling points with restarts
- [ ] Add batch strategy wrapper
- [ ] Refine default hyperparameters based on gridsearches
- [ ] How can we make restart wrappers be jittable (problem of non-static population sizes)?
- More strategies
    - [ ] Amalgalm-IDEA strategies
        - [ ] independent
        - [ ] bayesian
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
    - BIPOP-CMA-ES (`BIPOP_CMA_ES`)
    - IPOP-CMA-ES (`IPOP_CMA_ES`)
- Restart wrappers: 
    - Base restart class.
    - Simple reinit restart strategy.
    - BIPOP strategy with interleaved small/large populations.
    - IPOP strategy with growing population size.

##### Changed

- Both `ParamReshaper` and the rollout wrappers now support `pmap` over the population member dimension.
- Add `mean` state component to all strategies (also non-GD-based) for smoother evaluation protocols.
- Add `strategy_name` to all strategies.

##### Fixed

- Fix `BraxFitness` rollout wrapper and add train/test option.
- Fix small bug related to sigma decay in `Simple_GA`.

### [v0.0.3] - 21/02/2021

- Fix Python version requirements for evojax integration. Needs to be >3.6 since JAX versions stopped supporting 3.6.

### [v0.0.2] - 17/02/2021

- First public release including 11 strategies, rollout/network wrappers, utilities.

### [v0.0.1] - 22/11/2021

##### Added
- Adds all base functionalities
