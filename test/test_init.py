# Model packages
from openalea.rootcynaps import RootCyNAPS
# Utility packages
from openalea.fspm.utility.scenario import MakeScenarios as ms

def test_init():
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_06.xlsx", which=["Reference_Fischer"])
    for scenario_name, scenario in scenarios.items():
        root_cynaps = RootCyNAPS(time_step=3600, **scenario)

test_init()