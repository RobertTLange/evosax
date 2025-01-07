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
                    "Sep_CMA_ES",
                    "Full_iAMaLGaM",
                    "Indep_iAMaLGaM",
                    "MA_ES",
                    "LM_MA_ES",
                    "RmES",
                    "GLD",
                    "xNES",
                    "SNES",
                    "ESMC",
                    "DES",
                    "SAMR_GA",
                    "GESMR_GA",
                    "GuidedES",
                    "ASEBO",
                    "CR_FM_NES",
                    "MR15_GA",
                    "RandomSearch",
                    "LES",
                    "LGA",
                    "NoiseReuseES",
                    "HillClimber",
                    "EvoTF_ES",
                    "DiffusionEvolution",
                    "SV_CMA_ES",
                ],
            )
        else:
            metafunc.parametrize("strategy_name", ["LGA"])

    if "classic_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "classic_name",
                [
                    "Sphere",
                    "EllipsoidalOriginal",
                    "RastriginOriginal",
                    "BuecheRastrigin",
                    "LinearSlope",
                    # Part 2: Functions with low or moderate conditions
                    "AttractiveSector",
                    "StepEllipsoidal",
                    "RosenbrockOriginal",
                    "RosenbrockRotated",
                    # Part 3: Functions with high conditioning and unimodal
                    "EllipsoidalRotated",
                    "Discus",
                    "BentCigar",
                    "SharpRidge",
                    "DifferentPowers",
                    # Part 4: Multi-modal functions with adequate global structure
                    "RastriginRotated",
                    "Weierstrass",
                    "SchaffersF7",
                    "SchaffersF7IllConditioned",
                    "GriewankRosenbrock",
                    # Part 5: Multi-modal functions with weak global structure
                    "Schwefel",
                    "Lunacek",
                    "Gallagher101Me",
                    "Gallagher21Hi",
                    # "Katsuura",
                ],
            )
        else:
            metafunc.parametrize("classic_name", ["Sphere"])

    if "env_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "env_name",
                [
                    "CartPole-v1",
                ],
            )
        else:
            metafunc.parametrize("env_name", ["CartPole-v1"])
