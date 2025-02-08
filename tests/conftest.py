from evosax import Strategies
from evosax.problems.bbob import bbob_fns


def pytest_addoption(parser):
    parser.addoption("--all", action="store_true", help="run all combinations")


def pytest_generate_tests(metafunc):
    if "strategy_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "strategy_name",
                Strategies.keys(),
            )
        else:
            metafunc.parametrize("strategy_name", ["LGA"])

    if "classic_name" in metafunc.fixturenames:
        if metafunc.config.getoption("all"):
            metafunc.parametrize(
                "classic_name",
                bbob_fns.keys(),
            )
        else:
            metafunc.parametrize("classic_name", ["sphere"])

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
