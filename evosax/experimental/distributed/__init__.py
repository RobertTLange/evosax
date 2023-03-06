from .dist_open_es import DistributedOpenES


DistributedStrategies = {"DistributedOpenES": DistributedOpenES}


__all__ = ["DistributedOpenES", "DistributedStrategies"]
