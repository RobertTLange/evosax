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
                    "Augmented_RS"
                    # "PBT_ES",
                    # "Persistent_ES",
                    # "xNES",
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

    if "gymnax_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "gymnax_name",
                [
                    "CartPole-v1",
                ],
            )
        else:
            metafunc.parametrize("gymnax_name", ["CartPole-v1"])

    if "brax_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "brax_name",
                [
                    "ant",
                ],
            )
        else:
            metafunc.parametrize("brax_name", ["ant"])
