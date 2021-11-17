from typing import Union

import pandas as pd
import numpy as np

class EvaluateModel():
    def __init__(self, target_col: str, receipt_col: str):
        """
        Class used to produce all evaluation metrics
        """
        self.target_col = target_col
        self.receipt_col = receipt_col
        self._metrics = {}
        
    def fit(self, X: pd.DataFrame, reference: pd.DataFrame,
            probs: Union[pd.Series, np.ndarray]):
        """
        Calcualtes all metrics
        """
        summary = pd.Series(probs, name="prob").to_frame()
        summary[self.target_col] = reference[self.target_col].values
        summary["prob_bin"] = pd.cut(summary["prob"], bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01], include_lowest=True, right=False)
        self._metrics["summary"] = summary.groupby("prob_bin").aggregate({self.target_col: ["sum", "count", "mean"]})
        
        
        success = reference.copy()
        success["prob"] = probs
        success = pd.concat([X.reset_index(drop=True), success.reset_index(drop=True)], axis=1)
        
        result = []
        for r_id, df in success.groupby(self.receipt_col):
            max_prob = np.max(df["prob"])

            mask = df["prob"] == max_prob
            idx = np.where(mask)[0][0]
            df_max_prob = df[mask]

            max_dup_vars = (df_max_prob[X.columns].duplicated().max()) and (df_max_prob[self.target_col].max() == 1)

            result.append({self.receipt_col: r_id,
                           "obs_target": df[self.target_col].max(),
                           "max_dup_vars": int(max_dup_vars),
                           "sucess": int(df[self.target_col].values[idx] == 1)})
        result = pd.DataFrame(result)
        
        self._metrics["user_success_rate"] = result.sucess.mean()
        self._metrics["max_success_rate"] = result.obs_target.mean()
        self._metrics["norm_success_rate"] = result[result["obs_target"] == 1].sucess.mean()
        self._metrics["norm_no_dup_success_rate"] = result[(result["obs_target"] == 1) & (result["max_dup_vars"] == 0)].sucess.mean()