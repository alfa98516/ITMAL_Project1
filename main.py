import kagglehub
import time
import pandas as pd
import seaborn as sb
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
)
from sklearn.preprocessing import TargetEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.ensemble import RandomForestRegressor

path = kagglehub.dataset_download("debayank2024/house-price-prediction")

LR_copy = pd.read_csv(path + "/modified_data.csv")

print(LR_copy["price"].mean())
print((LR_copy["price"] > 25000000).sum())
# convert date to integer values so i can properly work with it
# not using year as the data is all within the same year
LR_copy["date"] = pd.to_datetime(LR_copy["date"])
LR_copy["month"] = LR_copy["date"].dt.month
LR_copy["day"] = LR_copy["date"].dt.day
LR_copy = LR_copy.drop(columns=["date"])
LR_copy["was_renovated"] = (LR_copy["yr_renovated"] > 0).astype(int)
LR_copy["effective_year"] = np.where(  # meaning "effective year it was renovated"
    LR_copy["yr_renovated"] > 0, LR_copy["yr_renovated"], LR_copy["yr_built"]
)
LR_copy["age_at_sale"] = LR_copy["yr_built"] - LR_copy["effective_year"]
LR_copy.drop(columns=["yr_renovated"])


LR_copy = LR_copy[
    LR_copy["price"] > 0
]  # Remove houses with 0 price value, these are actual outliers, as opposed to the very high value houses

RF_copy = LR_copy.copy()

encoded_LR_copy = pd.get_dummies(LR_copy, columns=["city", "statezip"], drop_first=True)

z_scaled_training_set = encoded_LR_copy.drop(
    columns=["price", "price_per_sqft", "street"]
)
for column in z_scaled_training_set.columns:
    z_scaled_training_set[column] = (
        z_scaled_training_set[column] - z_scaled_training_set[column].mean()
    ) / z_scaled_training_set[column].std()

print()
print(len(encoded_LR_copy.columns))
print(len(LR_copy.columns))
print()
features = z_scaled_training_set
# droping street as it is essentially a id, all are unique.
# price_per_sqft is just another copy of price, i can calculate it later if i need it
target = np.log1p(encoded_LR_copy.price)
price_bins = pd.qcut(target, q=10, labels=False, duplicates="drop")

(
    training_set_featuresLR,
    testing_set_featuresLR,
    training_set_targetLR,
    testing_set_targetLR,
) = train_test_split(
    features, target, test_size=0.2, random_state=985, stratify=price_bins
)  # 20% is the convention, might change as i do more testing


features_RF = RF_copy.drop(columns=["price", "price_per_sqft", "street"])
target_RF = np.log1p(RF_copy["price"])
(
    training_set_featuresRF,
    testing_set_featuresRF,
    training_set_targetRF,
    testing_set_targetRF,
) = train_test_split(
    features_RF, target_RF, test_size=0.2, random_state=985, stratify=price_bins
)

cat_cols = ["city", "statezip"]
cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=30)
prep = ColumnTransformer(
    [
        (
            "loc",
            TargetEncoder(cv=KFold(n_splits=5, shuffle=True, random_state=985)),
            cat_cols,
        )
    ],
    remainder="passthrough",
)
base_RF = RandomForestRegressor(n_estimators=300, random_state=985, n_jobs=-1)
rf_pipe = Pipeline(
    [
        ("prep", prep),
        ("model", base_RF),
    ]
)
base_XGB = XGBRegressor(random_state=985, n_jobs=-1)
xgb_pipe = Pipeline(
    [
        ("prep", prep),
        ("model", base_XGB),
    ]
)


LR = LinearRegression()
cross_validation_lr = cross_validate(
    LR, training_set_featuresLR, training_set_targetLR, cv=cv, scoring="r2"
)
print("mean:", cross_validation_lr["test_score"].mean())
print("std:", cross_validation_lr["test_score"].std())
print(
    "min/max:",
    cross_validation_lr["test_score"].min(),
    cross_validation_lr["test_score"].max(),
)


print("cross_val_lr: ", cross_validation_lr)
LR.fit(training_set_featuresLR, training_set_targetLR)

cross_validate(
    rf_pipe, training_set_featuresRF, training_set_targetRF, cv=cv, scoring="r2"
)

rf_pipe.fit(training_set_featuresRF, training_set_targetRF)
print(
    "Random Forest, initial score",
    rf_pipe.score(testing_set_featuresRF, testing_set_targetRF),
)
param_xgb = {
    # "model__n_estimators": [200, 400, 800],
    # "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    # "model__max_depth": [3, 4, 6, 8],
    # "model__min_child_weight": [1, 3, 5],
    # "model__subsample": [0.7, 0.85, 1.0],
    # "model__colsample_bytree": [0.5, 0.7, 1.0],
    # "model__reg_lambda": [1, 5, 10],
}

param_forest = {
    # "model__max_depth": [None, 10, 20, 30],
    # "model__min_samples_leaf": [1, 2, 4, 8],
    # "model__max_features": [0.3, 0.5, 0.7, 1.0],
}

# xgb_search = RandomizedSearchCV(xgb_pipe, param_xgb, n_iter=30, cv=5, random_state=985)
# xgb_search.fit(training_set_featuresRF, training_set_targetRF)
# print(xgb_search.best_params_, xgb_search.best_score_)

# random_forest_search = RandomizedSearchCV(
#     rf_pipe, param_forest, n_iter=20, cv=5, random_state=985
# )
# random_forest_search.fit(training_set_featuresRF, training_set_targetRF)
# print("forest best params: ", random_forest_search.best_params_)
# print("forest best score: ", random_forest_search.best_score_)


print("train LR:", LR.score(training_set_featuresLR, training_set_targetLR))
print("test LR:", LR.score(testing_set_featuresLR, testing_set_targetLR))

lr_cv = cross_validate(
    LR, training_set_featuresLR, training_set_targetLR, cv=cv, scoring="r2"
)
start_rf = time.time()
rf_cv = cross_validate(
    # random_forest_search.best_estimator_,
    rf_pipe,
    training_set_featuresRF,
    training_set_targetRF,
    cv=cv,
    scoring="r2",
)
end_rf = time.time()
start_xgb = time.time()
xgb_cv = cross_validate(
    # xgb_search.best_estimator_,
    xgb_pipe,
    training_set_featuresRF,
    training_set_targetRF,
    cv=cv,
    scoring="r2",
)
end_xgb = time.time()
print("one rf cv: ", end_rf - start_rf)
print("one xgb cv: ", end_xgb - start_xgb)


diff_rf = rf_cv["test_score"] - lr_cv["test_score"]
diff_xgb = xgb_cv["test_score"] - lr_cv["test_score"]
print("XGB:", xgb_cv["test_score"].mean(), "+/-", xgb_cv["test_score"].std())
print("LR:", lr_cv["test_score"].mean(), "+/-", lr_cv["test_score"].std())
print("RF:", rf_cv["test_score"].mean(), "+/-", rf_cv["test_score"].std())
print("mean diff (RF - LR):", diff_rf.mean(), "| std of diff:", diff_rf.std())
print("RF wins in", (diff_rf > 0).mean() * 100, "% of folds")

print("mean diff (XGB - LR):", diff_xgb.mean(), "| std of diff:", diff_xgb.std())
print("xgb wins in", (diff_xgb > 0).mean() * 100, "% of folds")
preds_log = LR.predict(testing_set_featuresLR)
preds_dollars = np.expm1(preds_log)
actual_dollars = np.expm1(testing_set_targetLR)

print("MAPE:", mean_absolute_percentage_error(actual_dollars, preds_dollars) * 100, "%")
print("MAE ($):", mean_absolute_error(actual_dollars, preds_dollars))
print("RMSE ($):", root_mean_squared_error(actual_dollars, preds_dollars))
