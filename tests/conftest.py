def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "strategy_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "strategy_name",
                [
                    "SimpleGA",
                    "SimpleES",
                    "CMA_ES",
                    "DE",
                    "PSO",
                    "OpenES",
                    "PGPE",
                    "ARS",
                    "PBT",
                    "PersistentES",
                    "xNES",
                    "Sep_CMA_ES",
                    "Full_iAMaLGaM",
                    "Indep_iAMaLGaM",
                    "MA_ES",
                    "LM_MA_ES",
                    "RmES",
                    "GLD",
                ],
            )
        else:
            metafunc.parametrize("strategy_name", ["PGPE"])

    if "classic_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "classic_name",
                [
                    "rosenbrock",
                    "quadratic",
                    "ackley",
                    "griewank",
                    "rastrigin",
                    "schwefel",
                    "himmelblau",
                    "six-hump",
                ],
            )
        else:
            metafunc.parametrize("classic_name", ["rosenbrock"])

    if "env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "env_name",
                ["CartPole-v1", "ant"],
            )
        else:
            metafunc.parametrize("env_name", ["CartPole-v1"])
