"""Distribution-based algorithms module."""

from .ars import ARS
from .asebo import ASEBO
from .bipop_cma_es import BIPOP_CMA_ES
from .cma_es import CMA_ES
from .cr_fm_nes import CR_FM_NES
from .discovered_es import DiscoveredES
from .esmc import ESMC
from .evotf_es import EvoTF_ES
from .gradientless_descent import GradientlessDescent
from .guided_es import GuidedES
from .hill_climbing import HillClimbing
from .iamalgam_full import iAMaLGaM_Full
from .iamalgam_univariate import iAMaLGaM_Univariate
from .ipop_cma_es import IPOP_CMA_ES
from .learned_es import LearnedES
from .lm_ma_es import LM_MA_ES
from .ma_es import MA_ES
from .noise_reuse_es import NoiseReuseES
from .open_es import Open_ES
from .persistent_es import PES
from .pgpe import PGPE
from .random_search import RandomSearch
from .rm_es import Rm_ES
from .sep_cma_es import Sep_CMA_ES
from .simple_es import SimpleES
from .simulated_annealing import SimulatedAnnealing
from .snes import SNES
from .sv.sv_cma_es import SV_CMA_ES
from .sv.sv_open_es import SV_Open_ES
from .xnes import xNES

distribution_based_algorithms = {
    "ARS": ARS,
    "ASEBO": ASEBO,
    "BIPOP_CMA_ES": BIPOP_CMA_ES,
    "CMA_ES": CMA_ES,
    "CR_FM_NES": CR_FM_NES,
    "DES": DiscoveredES,
    "ESMC": ESMC,
    "EvoTF_ES": EvoTF_ES,
    "GradientlessDescent": GradientlessDescent,
    "GuidedES": GuidedES,
    "HillClimbing": HillClimbing,
    "iAMaLGaM_Full": iAMaLGaM_Full,
    "iAMaLGaM_Univariate": iAMaLGaM_Univariate,
    "IPOP_CMA_ES": IPOP_CMA_ES,
    "LES": LearnedES,
    "LM_MA_ES": LM_MA_ES,
    "MA_ES": MA_ES,
    "NoiseReuseES": NoiseReuseES,
    "Open_ES": Open_ES,
    "PES": PES,
    "PGPE": PGPE,
    "RandomSearch": RandomSearch,
    "Rm_ES": Rm_ES,
    "Sep_CMA_ES": Sep_CMA_ES,
    "SimulatedAnnealing": SimulatedAnnealing,
    "SimpleES": SimpleES,
    "SNES": SNES,
    "SV_CMA_ES": SV_CMA_ES,
    "SV_Open_ES": SV_Open_ES,
    "xNES": xNES,
}

__all__ = list(distribution_based_algorithms.keys())
