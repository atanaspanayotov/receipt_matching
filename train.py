import pickle
import time
import argparse
import logging
from collections import Counter

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, GroupShuffleSplit
from xgboost import XGBClassifier

from src.custom_selector import ColumnSelector
from src.evaluate import EvaluateModel

def initialize_logger(logger: logging.Logger,
                      log_level: int = logging.INFO,
                      log_format: str =
                      '%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
    logger.setLevel(level=log_level)
    formatter = logging.Formatter(log_format)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

logger = logging.getLogger(__name__)
initialize_logger(logger=logger)

def train(model_file: str,
          train_file: str,
          random_state: int):
    """
    Train receipt matching model
    """
    data = pd.read_csv(train_file, delimiter=":")
    
    logger.info(f"Read {train_file} with {data.shape[0]} rows")

    matched_tx_col = "matched_transaction_id"
    feature_tx_col = "feature_transaction_id"
    receipt_id = 'receipt_id'
    target_col = 'target'
    features = ['DateMappingMatch', 'TimeMappingMatch', 'ShortNameMatch', 'DescriptionMatch', 'PredictedNameMatch',
                'PredictedTimeCloseMatch', 'PredictedAmountMatch', 'AmountMappingMatch', 'DifferentPredictedDate',
                'DifferentPredictedTime']
    
    data[target_col] = (data[matched_tx_col] == data[feature_tx_col]).astype(int)

    group_kfold = GroupShuffleSplit(test_size=0.25, random_state=random_state)
    groups = data[receipt_id]
    train_index, test_index = list(group_kfold.split(data.values, None, groups))[0]

    X_train = data.iloc[train_index, :][features]
    y_train = data.iloc[train_index, :][target_col]
    group_col_train = data.iloc[train_index, :][receipt_id]
    X_test = data.iloc[test_index, :][features]
    y_test = data.iloc[test_index, :][target_col]

    logger.info(f"Training data size = {X_train.shape[0]}")
    logger.info(f"Test data size = {X_test.shape[0]}")

    counter = Counter(y_train)
    scale_pos_weight = counter[0] / counter[1]
    model = Pipeline(steps=[("sel", ColumnSelector(corr_threshold=0.90)),
                            ("cls", XGBClassifier(scale_pos_weight=scale_pos_weight,
                                                   random_state=random_state))])
    
    logger.info(f"Start Hyperaparamter Tuning Phase...")
    not_fixed_params = {
        "cls__gamma": [0, 0.5, 1, 3, 5],
        "cls__max_depth": [3, 5, 10, 20],
        "cls__learning_rate": [0.01, 0.1, 0.2],
        "cls__reg_alpha": [0, 0.5, 1, 3, 5],
        "cls__min_child_weight": [10, 20, 30, 40],
        "cls__reg_lambda": [0, 0.5, 1, 3, 5],
        "cls__n_estimators": [10, 15, 20, 30, 40]
    }
    cv = list(GroupKFold(n_splits=4).split(X_train, y_train, group_col_train))
    optimizer = RandomizedSearchCV(model,
                                   param_distributions=not_fixed_params,
                                   n_iter=300,
                                   scoring="roc_auc",
                                   n_jobs=-1,
                                   cv=cv,
                                   random_state=random_state,
                                   refit=True,
                                   verbose=1)
    optimizer = optimizer.fit(X_train, y_train)

    start_time = time.time()
    logger.info(f"Start Model Training...")
    model = model.set_params(**optimizer.best_params_)
    model.fit(X_train, y_train)
    logger.info(f"Model training took {round(time.time() - start_time, 2)} seconds")

    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]
    
    train_gini = roc_auc_score(y_train, pred_train) * 2 - 1
    test_gini = roc_auc_score(y_test, pred_test) * 2 - 1
    
    logger.info(f"Gini on train data = {round(train_gini, 4)}")
    logger.info(f"Gini on test data = {round(test_gini, 4)}")
    
    metrics = EvaluateModel(target_col, receipt_id)
    metrics.fit(X_test, data.iloc[test_index, :][[receipt_id, target_col]], pred_test)
    
    logger.info(metrics._metrics["summary"])
    
    logger.info("====Model Success Rate====")
    logger.info(f"Total Success Rate = {round(metrics._metrics['user_success_rate'], 4)}")
    logger.info(f"Maximum Success Rate = {round(metrics._metrics['max_success_rate'], 4)}")
    logger.info(f"Normalized For Imp. Success Rate = {round(metrics._metrics['norm_success_rate'], 4)}")
    logger.info(f"Normalized For Imp. and Dups. Success Rate = {round(metrics._metrics['norm_no_dup_success_rate'], 4)}")

    with open(model_file, 'wb') as fp:
        pickle.dump(model, fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('train_file')
    parser.add_argument('--random_state', default=21)
    args = parser.parse_args()
    
    train(args.model_file, args.train_file, random_state=args.random_state)