import pandas as pd
import os

class Reporter:
    def export(self, step_metrics, position_data, errors, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        pd.DataFrame(step_metrics).to_csv(os.path.join(output_dir, 'step_metrics.csv'), index=False)
        pd.DataFrame(position_data).to_csv(os.path.join(output_dir, 'position_data.csv'), index=False)
        pd.DataFrame(errors).to_csv(os.path.join(output_dir, 'errors.csv'), index=False)
