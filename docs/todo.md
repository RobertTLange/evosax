## TODOs, Notes & Questions
- [ ] Refactor sampling - file with different relevant types
- [ ] Jit with frozen dicts? -> ES entire batches/pipelines
- [ ] TPU example/How to do pmap over devices/hosts => Run on GCP VM
- [ ] Add TPU to runtime benchmarks
- [ ] Clean up visualizations/animations + proper general API
- [ ] Implement more strategies
    - [ ] Add restarts for CMA-ES
    - [ ] Add NES strategy
    - [ ] Add PEPG strategy
- [ ] Implement more examples
    - [ ] Flax CNN MNIST classification example
    - [ ] Update cuda driver for conv operation?
    - [x] Small Haiku RNN example - meta on bandit
- [ ] Provide a nice wrapper to vmap reshape and eval
- [ ] Logging integration
    - [x] Save + load logger as pkl
    - [ ] Integrate with tensorboard?! Can't jit!

## Final steps
- [ ] Make pip installable
- [ ] Run full time benchmarks
- [ ] Full gym control examples?
- [ ] [Connect notebooks with example Colab](https://colab.research.google.com/github/googlecolab/colabtools/blob/master/notebooks/colab-github-demo.ipynb#scrollTo=K-NVg7RjyeTk)


## 24/01/21 - Clean up + Gymnax integration
- [x] Get rid of MLP reshaper = only single one for all packages!
- [x] Integrate `gymnax` into all RL examples with rollout wrapper

## 25/01/21 - Problems + Visualizer to `gymnax`
- [x] Move problems to gymnax (bandit + pendulum task)
- [x] Move visualizer to gymnax
- [x] Add save and load log functions
- [x] Update animate viz example notebook with reload

## 10/02/21 - Update collectors with new gymnax dojos
- Use `EvaluationDojo` + `MinimalEvaluationAgent`
- Need to figure out how to scan through RNN agent
- Weird problems with Conv operations and tensorflow
