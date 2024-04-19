from .snes import SNES
from .open_es import OpenES
from .pgpe import PGPE
from .sep_cma_es import Sep_CMA_ES


DistributedStrategies = {
    "SNES": SNES,
    "OpenES": OpenES,
    "PGPE": PGPE,
    "Sep_CMA_ES": Sep_CMA_ES,
}


__all__ = [
    "SNES",
    "OpenES",
    "PGPE",
    "Sep_CMA_ES",
    "DistributedStrategies",
]
