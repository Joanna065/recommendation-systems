import logging
import os
from datetime import datetime
from pathlib import Path

from src.datasets.funk_svd_dataset import ratings_funk_process
from src.experiments.collaborative_filtering.funk_svd_exp import (run_funk_svd_experiment,
                                                                  save_logs, save_model)
from src.settings import DATA_DIR, RESULT_DIR

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

# USER = "arktos8"
# USER = "LorethRex"
USER = "Nikki97"
USER_DATA_PATH = os.path.join(DATA_DIR, 'processed', f'user_{USER}')

LOGS_DIR = os.path.join(RESULT_DIR, 'logs', f'funk_svd/filmweb/{USER}')
Path(LOGS_DIR).mkdir(parents=True, exist_ok=True)
MODEL_DIR = os.path.join(RESULT_DIR, 'models', f'funk_svd/filmweb/{USER}')
Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)

date = datetime.today()
date_now = date.strftime('%Y-%m-%d_%H-%M')

params = {
    'latent_factors': 45,
    'lr': 0.005,
    'regularization': 0.02,
    'epochs': 100,
    'min_rate': 0.5,
    'max_rate': 5.0,
    'shuffle': True
}

# load data
ratings_concat = ratings_funk_process(os.path.join(USER_DATA_PATH, 'concat_ratings.csv'),
                                      with_timestamp=False)

model, model_dump_dict, logs_dict, trained_epochs = run_funk_svd_experiment(ratings_concat,
                                                                            params,
                                                                            train_split=0.8,
                                                                            val=True, seed=25)

exp_name = f'{date_now}_FunkSVD-{params["latent_factors"]}-factors-' \
           f'{params["lr"]}-lr-{params["regularization"]}-reg-' \
           f'{trained_epochs}-epochs-{params["shuffle"]}-shuffle'

save_model(MODEL_DIR, exp_name, model_dump_dict)
save_logs(LOGS_DIR, exp_name, logs_dict)
