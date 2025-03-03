"""Distribution-based algorithms module."""

from .ars import ARS
from .asebo import ASEBO
from .bipop_cma_es import BIPOP_CMA_ES
from .cma_es import CMA_ES
from .cr_fm_nes import CR_FM_NES
from .des import DES
from .esmc import ESMC
from .evotf_es import EvoTF_ES
from .gld import GLD
from .guided_es import GuidedES
from .hill_climber import HillClimber
from .iamalgam_full import iAMaLGaM_Full
from .iamalgam_univariate import iAMaLGaM_Univariate
from .ipop_cma_es import IPOP_CMA_ES
from .les import LES
from .lm_ma_es import LM_MA_ES
from .ma_es import MA_ES
from .noise_reuse_es import NoiseReuseES
from .open_es import Open_ES
from .pes import PES
from .pgpe import PGPE
from .random_search import RandomSearch
from .rm_es import Rm_ES
from .sep_cma_es import Sep_CMA_ES
from .sim_anneal import SimAnneal
from .simple_es import SimpleES
from .snes import SNES
from .sv_cma_es import SV_CMA_ES
from .sv_open_es import SV_Open_ES
from .xnes import xNES

distribution_based_algorithms = {
    "ARS": ARS,
    "ASEBO": ASEBO,
    "BIPOP_CMA_ES": BIPOP_CMA_ES,
    "CMA_ES": CMA_ES,
    "CR_FM_NES": CR_FM_NES,
    "DES": DES,
    "ESMC": ESMC,
    "EvoTF_ES": EvoTF_ES,
    "GLD": GLD,
    "GuidedES": GuidedES,
    "HillClimber": HillClimber,
    "iAMaLGaM_Full": iAMaLGaM_Full,
    "iAMaLGaM_Univariate": iAMaLGaM_Univariate,
    "IPOP_CMA_ES": IPOP_CMA_ES,
    "LES": LES,
    "LM_MA_ES": LM_MA_ES,
    "MA_ES": MA_ES,
    "NoiseReuseES": NoiseReuseES,
    "OpenES": Open_ES,
    "PES": PES,
    "PGPE": PGPE,
    "RandomSearch": RandomSearch,
    "Rm_ES": Rm_ES,
    "Sep_CMA_ES": Sep_CMA_ES,
    "SimAnneal": SimAnneal,
    "SimpleES": SimpleES,
    "SNES": SNES,
    "SV_CMA_ES": SV_CMA_ES,
    "SV_OpenES": SV_Open_ES,
    "xNES": xNES,
}

__all__ = list(distribution_based_algorithms.keys())
