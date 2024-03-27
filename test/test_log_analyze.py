import pickle

from root_cynaps.root_cynaps import Model
from data_utility.log import Logger
#from data_utility.data_analysis import analyze_data

def test_log_analyze():
    with open('test/inputs/root00719.pckl', 'rb') as f:
        g = pickle.load(f)
    
    root_cynaps = Model(g=g, time_step=3600)
    logger = Logger(model_instance=root_cynaps, outputs_dirpath="test/outputs", 
                    time_step_in_hours=1,
                    logging_period_in_hours=1,
                    recording_sums=False,
                    recording_raw=False,
                    recording_images=True, plotted_property="Nm", 
                    recording_mtg=False,
                    recording_performance=False,
                    echo=True)
    

    for step in range(5):
        # Placed here also to capture mtg initialization
        logger()

        root_cynaps.run()

    logger.stop()
    #analyze_data(outputs_dirpath="test/outputs")

test_log_analyze()