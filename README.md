# Yolanda

This repository contains the codebase for my solution to the MeLi data challenge 2021.

## TL;DR

The final model consists of two stacked models: the **level-1** model is a single LightGBM trained to forecast the sold quantity, and **level-2** model is a neural network that takes **level-1** output as input and learns to minimize the RPS.

## Submission generation

All the codes for data preparation and modeling are in the `notebooks/` directory, however, only a subset of these notebooks are used to generate the final submission. Below is the list of the utilized notebooks as well as the order of execution:

**Data preparation:**

- Download and place the competition data in the `data/` directory.
- Run `notebooks/preproc-m1.ipynb` -> prepares data for level-1 model.
- Run `notebooks/eda.ipynb` -> performs EDA, computes `unpredictable.csv` (skus with not enough information) and computes `scales.csv` (scale values for RMSSE).
- Run `notebooks/preproc_assessment.ipynb` -> computes `skus_assess_m1.yaml` (skus for assessment of level-1 model), the synthetic validation sets (`validation_seed*.csv` & `validation_seed*\_harder.csv`) and `validation_m3.csv` (the targets for the level-2 model).

**Modeling:**

- Run `notebooks/encoders.ipynb` -> trains the categorical encoders (GLMM) for stage1 and stage2 of level-1 model.
- Run `notebooks/train_lgbm-m1-sm.ipynb` -> train a LightGBM model for predicting the sold quantity at sku level. The stage1 is for generating the oof predictions, while stage2 is for generating the final predictions.
  - output-1: `results/oof_preds_lgbm-m1.csv` the oof predictions (for the period of 2021-03-02 to 2021-03-31).
  - output-2: `results/preds_m1_lgbm_sub{SUB_NBR}.csv` the predictions for the test period.
- Run `notebooks/train_1dcnn-m3.ipynb` -> take the oof predictions of level1 model and train a 1dcnn stacking model that learns to minimze the RPS. Then applies the trained model over the level-1 predictions on the test period.
