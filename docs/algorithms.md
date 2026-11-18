# Algorithms

evosax implements 37 algorithms in two families. All of them share the
[ask-eval-tell API](guides/concepts.md) and can be imported from `evosax.algorithms`:

```python
from evosax.algorithms import CMA_ES, DifferentialEvolution
```

The class name links to the API reference of each algorithm.

## Distribution-based

These algorithms maintain a search distribution and are initialized from a starting
mean with `init(key, mean, params)`.

| Algorithm | Class | Reference |
| --- | --- | --- |
| Simple ES | [`SimpleES`](reference/algorithms/simple_es.md) | [Rechenberg (1973)](https://onlinelibrary.wiley.com/doi/abs/10.1002/fedr.19750860506) |
| OpenAI-ES | [`Open_ES`](reference/algorithms/open_es.md) | [Salimans et al. (2017)](https://arxiv.org/abs/1703.03864) |
| CMA-ES | [`CMA_ES`](reference/algorithms/cma_es.md) | [Hansen & Ostermeier (2001)](https://arxiv.org/abs/1604.00772) |
| Sep-CMA-ES | [`Sep_CMA_ES`](reference/algorithms/sep_cma_es.md) | [Ros & Hansen (2008)](https://hal.inria.fr/inria-00287367/document) |
| xNES | [`xNES`](reference/algorithms/xnes.md) | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) |
| SNES | [`SNES`](reference/algorithms/snes.md) | [Wierstra et al. (2014)](https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf) |
| MA-ES | [`MA_ES`](reference/algorithms/ma_es.md) | [Bayer & Sendhoff (2017)](https://ieeexplore.ieee.org/document/7875115) |
| LM-MA-ES | [`LM_MA_ES`](reference/algorithms/lm_ma_es.md) | [Loshchilov et al. (2017)](https://arxiv.org/abs/1705.06693) |
| Rm-ES | [`Rm_ES`](reference/algorithms/rm_es.md) | [Li & Zhang (2017)](https://ieeexplore.ieee.org/document/8080257) |
| PGPE | [`PGPE`](reference/algorithms/pgpe.md) | [Sehnke et al. (2010)](https://link.springer.com/chapter/10.1007/978-3-540-87536-9_40) |
| ARS | [`ARS`](reference/algorithms/ars.md) | [Mania et al. (2018)](https://arxiv.org/abs/1803.07055) |
| ESMC | [`ESMC`](reference/algorithms/esmc.md) | [Merchant et al. (2021)](https://arxiv.org/abs/2107.09661) |
| Persistent ES | [`PersistentES`](reference/algorithms/persistent_es.md) | [Vicol et al. (2021)](https://arxiv.org/abs/2112.13835) |
| Noise-Reuse ES | [`NoiseReuseES`](reference/algorithms/noise_reuse_es.md) | [Li et al. (2023)](https://arxiv.org/abs/2304.12180) |
| CR-FM-NES | [`CR_FM_NES`](reference/algorithms/cr_fm_nes.md) | [Nomura & Ono (2022)](https://arxiv.org/abs/2201.11422) |
| Guided ES | [`GuidedES`](reference/algorithms/guided_es.md) | [Maheswaranathan et al. (2018)](https://arxiv.org/abs/1806.10230) |
| ASEBO | [`ASEBO`](reference/algorithms/asebo.md) | [Choromanski et al. (2019)](https://arxiv.org/abs/1903.04268) |
| Discovered ES | [`DiscoveredES`](reference/algorithms/discovered_es.md) | [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260) |
| Learned ES | [`LearnedES`](reference/algorithms/learned_es.md) | [Lange et al. (2023a)](https://arxiv.org/abs/2211.11260) |
| CMA-anchored LES | [`CMAAnchoredLES`](reference/algorithms/cma_anchored_les.md) | Lange et al. (2026) |
| EvoTF | [`EvoTF_ES`](reference/algorithms/evotf_es.md) | [Lange et al. (2024)](https://arxiv.org/abs/2403.02985) |
| iAMaLGaM-Full | [`iAMaLGaM_Full`](reference/algorithms/iamalgam_full.md) | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf) |
| iAMaLGaM-Univariate | [`iAMaLGaM_Univariate`](reference/algorithms/iamalgam_univariate.md) | [Bosman et al. (2013)](https://homepages.cwi.nl/~bosman/publications/2013_benchmarkingparameterfree.pdf) |
| Gradientless Descent | [`GradientlessDescent`](reference/algorithms/gradientless_descent.md) | [Golovin et al. (2019)](https://arxiv.org/abs/1911.06317) |
| Simulated Annealing | [`SimulatedAnnealing`](reference/algorithms/simulated_annealing.md) | [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759) |
| Hill Climbing | [`HillClimbing`](reference/algorithms/hill_climbing.md) | [Rasdi Rere et al. (2015)](https://www.sciencedirect.com/science/article/pii/S1877050915035759) |
| Random Search | [`RandomSearch`](reference/algorithms/random_search.md) | [Bergstra & Bengio (2012)](https://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf) |
| SV-CMA-ES | [`SV_CMA_ES`](reference/algorithms/sv_cma_es.md) | [Braun et al. (2024)](https://arxiv.org/abs/2410.10390) |
| SV-OpenAI-ES | [`SV_Open_ES`](reference/algorithms/sv_open_es.md) | [Liu et al. (2017)](https://arxiv.org/abs/1704.02399) |

## Population-based

These algorithms maintain an explicit population and are initialized from an evaluated
population with `init(key, population, fitness, params)`.

| Algorithm | Class | Reference |
| --- | --- | --- |
| Simple GA | [`SimpleGA`](reference/algorithms/simple_ga.md) | [Such et al. (2017)](https://arxiv.org/abs/1712.06567) |
| MR15-GA | [`MR15_GA`](reference/algorithms/mr15_ga.md) | [Rechenberg (1978)](https://link.springer.com/chapter/10.1007/978-3-642-81283-5_8) |
| SAMR-GA | [`SAMR_GA`](reference/algorithms/samr_ga.md) | [Clune et al. (2008)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1000187) |
| GESMR-GA | [`GESMR_GA`](reference/algorithms/gesmr_ga.md) | [Kumar et al. (2022)](https://arxiv.org/abs/2204.04817) |
| Learned GA | [`LearnedGA`](reference/algorithms/learned_ga.md) | [Lange et al. (2023b)](https://arxiv.org/abs/2304.03995) |
| Diffusion Evolution | [`DiffusionEvolution`](reference/algorithms/diffusion_evolution.md) | [Zhang et al. (2024)](https://arxiv.org/abs/2410.02543) |
| Differential Evolution | [`DifferentialEvolution`](reference/algorithms/differential_evolution.md) | [Storn & Price (1997)](https://link.springer.com/article/10.1023/A:1008202821328) |
| Particle Swarm Optimization | [`PSO`](reference/algorithms/pso.md) | [Kennedy & Eberhart (1995)](https://ieeexplore.ieee.org/document/488968) |

## Learned algorithms

`LearnedES`, `LearnedGA`, `EvoTF_ES` and `CMAAnchoredLES` use meta-trained neural
network components and load pretrained weights that ship with evosax. `DiscoveredES`
comes from the same line of work and needs no pretrained weights.

The implementations of Learned ES and Discovered ES are independent reimplementations
and are unrelated to Google DeepMind.
