# Model packages
from root_cynaps.root_cynaps import Model
# Utility packages
from initialize.initialize import MakeScenarios as ms

def test_init():
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_06.xlsx", which=["Reference_Fischer"])
    for scenario_name, scenario in scenarios.items():
        root_cynaps = Model(time_step=3600, **scenario)

test_init()