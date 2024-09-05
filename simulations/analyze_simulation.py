# Utility packages
from analyze.analyze import analyze_data
from log.logging import Logger


if __name__ == '__main__':

    scenarios = ["Drew_1975_low"]

    analyze_data(scenarios=scenarios, outputs_dirpath="outputs", target_properties=None, **Logger.heavy_log)
