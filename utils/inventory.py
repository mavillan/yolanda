import numpy as np
import pandas as pd
from scipy import interpolate
from scipy import stats
from tqdm import tqdm

class InventoryDaysPredictor():
    """
    Class to transform point predictions into probability 
    distributions over the time range.
    """

    def __init__(self, train):
        self.train = train.copy(deep=True)

    def fit(self, preds):
        parameters = self.train.groupby('sku').agg({'y':['mean', 'std']}).y
        # replace nan means by overall mean
        idx = parameters[parameters["mean"].isna()].index
        parameters.loc[idx, "mean"] = parameters["mean"].mean()
        # replace zero means by overall mean
        idx = parameters[parameters["mean"] == 0].index
        parameters.loc[idx, "mean"] = parameters["mean"].mean()
        
        # replace nan stds by overall std
        idx = parameters[parameters["std"].isna()].index
        parameters.loc[idx,"std"] = parameters["std"].mean()
        # replace nan stds by overall std
        idx = parameters[parameters["std"] == 0].index
        parameters.loc[idx,"std"] = parameters["std"].mean()
        
        self.parameters = parameters.to_dict()

        predictors = dict()
        days = np.arange(1,31)
        for sku,df in tqdm(preds.groupby("sku")):
            cumpred = df.y_pred.values.cumsum()
            interp = interpolate.interp1d(cumpred, days, bounds_error=False, fill_value=(-np.inf,np.inf))
            predictors[sku] = interp
        self.predictors = predictors

    def predict(self, sku, stock):
        mean = self.parameters['mean'][sku]
        std = self.parameters['std'][sku]
        days_to_stockout = float(np.clip(self.predictors[sku](stock), a_min=1, a_max=30))
        std_days = std/mean
        return days_to_stockout,std_days

    def predict_proba(self, sku, stock, dist_kwargs, lambda1, lambda2):
        days_to_stockout,std_days = self.predict(sku, stock)
        scale = std_days * (lambda1*(days_to_stockout**lambda2))
        days = np.arange(1,31)
        probs = stats.gennorm.pdf(days, loc=days_to_stockout, scale=scale, **dist_kwargs)
        #if prob is zero, replace with uniform
        if np.sum(probs) == 0: return np.ones(30) / 30
        return probs/np.sum(probs)
    
    
class IDP():
    """
    Class to transform oof predictions into
    inventory days predictions
    """

    def __init__(self):
        pass

    def fit(self, preds):
        predictors = dict()
        days = np.arange(1,31)
        for sku,df in tqdm(preds.groupby("sku")):
            cumpred = df.y_pred.values.cumsum()
            interp = interpolate.interp1d(cumpred, days, bounds_error=False, fill_value=(-np.inf,np.inf))
            predictors[sku] = interp
        self.predictors = predictors

    def predict(self, sku, stock):
        idp = float(self.predictors[sku](stock))
        idp_clip = float(np.clip(idp, a_min=1, a_max=30))
        return idp,idp_clip
