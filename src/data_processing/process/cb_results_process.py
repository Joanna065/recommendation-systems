import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklego.pandas_utils import log_step


@log_step(level=logging.DEBUG)
def build_spearman_drop_frame(results_dict: Dict, drop_rate: float):
    spearman_results = [spearman for spearman, p_val in
                        results_dict[f'drop_{drop_rate}']['spearman']]
    spearman_std = np.std(np.abs(spearman_results))
    spearman_mean = np.mean(np.abs(spearman_results))

    df = pd.DataFrame()
    df['spearman_correlation'] = spearman_results
    df['drop_rate'] = [f'{drop_rate}, mean = {spearman_mean:.4f} '
                       f'± {spearman_std:.4f}'] * len(spearman_results)

    return df
