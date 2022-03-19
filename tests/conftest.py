def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "strategy_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "strategy_name",
                [
                    "Simple_GA",
                    "Simple_ES",
                    "CMA_ES",
                    "Differential_ES",
                    "PSO_ES",
                    "Open_ES",
                    "PGPE_ES",
                    "Augmented_RS",
                    "PBT_ES",
                    "Persistent_ES",
                    "xNES",
                    "Sep_CMA_ES",
                    "Full_iAMaLGaM",
                ],
            )
        else:
            metafunc.parametrize("strategy_name", ["CMA_ES"])

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
