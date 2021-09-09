import numpy as np
import pandas as pd
import gc

class Featurador():

    def __init__(self, raw):
        self.dataframe = raw.copy()

    def fit(self, left_limit, right_limit, site=None):
        query_str = "@left_limit <= date <= @right_limit"
        if site is not None:
            query_str += " & site_id == @site"
        dataframe = self.dataframe.query(query_str).reset_index(drop=True)
        dataframe["tmp"] = dataframe.eval("sold_quantity*minutes_active")
        q_mean = (dataframe.groupby("sku")["tmp"].sum() / dataframe.groupby("sku")["minutes_active"].sum()).reset_index(name="q_mean")
        dataframe = dataframe.merge(q_mean, how="inner", on="sku")
        dataframe.drop("tmp", axis=1, inplace=True)

        dataframe["tmp"]  = dataframe.eval("minutes_active * (sold_quantity - q_mean)**2")
        q_std = (dataframe.groupby("sku")["tmp"].sum() / dataframe.groupby("sku")["minutes_active"].sum()).reset_index(name="q_std")
        dataframe = dataframe.merge(q_std, how="inner", on="sku")
        dataframe.drop("tmp", axis=1, inplace=True)

        # imputation of for skus with no value
        q_mean_imp = (
            dataframe
            .loc[:,["sku","item_domain_id","q_mean","q_std"]]
            .drop_duplicates()
            .query("q_mean > 0")
            .groupby("item_domain_id")["q_mean"]
            .mean()
            .reset_index(name="q_mean_imp")
        )
        q_std_imp = (
            dataframe
            .loc[:,["sku","item_domain_id","q_mean","q_std"]]
            .drop_duplicates()
            .query("q_std > 0")
            .groupby("item_domain_id")["q_std"]
            .median()
            .reset_index(name="q_std_imp")
        )

        del dataframe
        gc.collect()

        self.q_mean = q_mean
        self.q_mean_imp = q_mean_imp
        self.q_std = q_std
        self.q_std_imp = q_std_imp

    def transform(self, dataframe):
        dataframe = dataframe.merge(self.q_mean, how="inner", on="sku")
        dataframe = dataframe.merge(self.q_std, how="inner", on="sku")

        dataframe = (
            dataframe
            .merge(self.q_mean_imp, how="left", on="item_domain_id")
            .merge(self.q_std_imp, how="left", on="item_domain_id")
        )

        idx = dataframe[dataframe.q_mean_imp.isna()].index
        dataframe.loc[idx, "q_mean_imp"] = self.q_mean_imp.q_mean_imp.median()
        idx = dataframe[dataframe.q_std_imp.isna()].index
        dataframe.loc[idx, "q_std_imp"] = self.q_std_imp.q_std_imp.median()

        idx = dataframe[dataframe.q_mean == 0].index
        dataframe.loc[idx, "q_mean"] = dataframe.loc[idx, "q_mean_imp"]
        idx = dataframe[dataframe.q_std == 0].index
        dataframe.loc[idx, "q_std"] = dataframe.loc[idx, "q_std_imp"]
        
        dataframe.drop(["q_mean_imp","q_std_imp"], axis=1, inplace=True)
        return dataframe
