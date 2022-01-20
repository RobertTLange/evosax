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
                    "Open_NES",
                    "PEPG_ES",
                    # "PBT_ES",
                    # "Persistent_ES",
                    # "xNES",
                    # "Augmented_RS"
                ],
            )
        else:
            metafunc.parametrize("strategy_name", ["CMA_ES"])
