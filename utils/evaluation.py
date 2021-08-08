import numpy as np
import pandas as pd

class RMSSE(self, valid_dataframe, scales_dataframe):
    self.valid_dataframe = valid_dataframe
    self.scales_dataframe = scales_dataframe

def _evaluate(self, predictions):
    valid_dataframe = self.valid_dataframe.copy()
    valid_dataframe["ypred"] = predictions
    valid_dataframe["sq_error"] = valid_dataframe.eval("(y-ypred)**2")
    mse = valid_dataframe.groupby("sku")["sq_error"].mean().reset_index(name="mse")
    mrg = pd.merge(mse, self.scales_dataframe, how="inner", on="sku")
    return mrg.eval("sqrt(mse)/scale").mean()

def evaluate(self, ypred, dtrain):
    metric = self._evaluate(ypred)
    return "rmsse", metric, False
