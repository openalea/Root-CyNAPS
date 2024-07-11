# Model packages
from root_cynaps.root_cynaps import Model
# Utility packages
from initialize.initialize import MakeScenarios as ms


def single_run(scenario, simulation_length=2500):
    root_cynaps = Model(time_step=3600, **scenario)

    for _ in range(simulation_length):
        root_cynaps.run()
        

def test_run(simulation_length=1):
    scenarios = ms.from_table(file_path="inputs/Scenarios_24_06.xlsx", which=["Reference_Fischer"])
    
    for scenario_name, scenario in scenarios.items():
        print(f"[INFO] Launching scenario {scenario_name}...")
        single_run(scenario=scenario, simulation_length=simulation_length)
        
if __name__ == "__main__":
    test_run()