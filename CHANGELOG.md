**v0.0.1**
- **`strategies`**:
    - Base Gaussian Strategy
    - CMA-ES
    - OpenAI NES
- **`utils`**:
    - Logger for ES
    - Fitness reshaping/normalisation
    - 'Flat-to-Net' reshaper for going from `ask` to `eval` step
- **`notebooks`**:
    - Gaussian ES on low dim. problems (Rosenbrock, etc.)
    - CMA-ES on Pendulum task
    - OpenAI NES on CNN for MNIST
    - CMA-ES on Bandit Meta-LSTM


- This way we can easily `vmap` across parameter configurations.
- Check out Flax surgery setup: https://flax.readthedocs.io/en/latest/howtos/model_surgery.html
  - `flatten_dict` and `unflatten_dict` for network modules - jittable?!
- Check out Flax ensembling via pmap: https://flax.readthedocs.io/en/latest/howtos/ensembling.html
