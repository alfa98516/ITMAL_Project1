import kagglehub
import pandas as pd
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import ShuffleSplit

path = kagglehub.dataset_download("debayank2024/house-price-prediction")

knn_copy = pd.read_csv(path + "/modified_data.csv")

print(knn_copy["price"].describe())
print(knn_copy["price"].sort_values(ascending=False).head(20))
print((knn_copy["price"] == 0).sum())

sb.histplot(knn_copy["price"], bins=50)
plt.show()

# convert date to integer values so i can properly work with it
# not using year as the data is all within the same year
knn_copy["date"] = pd.to_datetime(knn_copy["date"])
knn_copy["month"] = knn_copy["date"].dt.month
knn_copy["day"] = knn_copy["date"].dt.day
knn_copy = knn_copy.drop(columns=["date"])

knn_copy = knn_copy[
    knn_copy["price"] > 0
]  # Remove houses with 0 price value, these are actual outliers, as opposed to the very high value houses
print(len(knn_copy.columns))

print((knn_copy["price"] == 0).sum())
print(knn_copy["price"].sort_values().head(20))


encoded_knn_copy = pd.get_dummies(
    knn_copy, columns=["city", "statezip"], drop_first=True
)

print(len(encoded_knn_copy.columns))
z_scaled_training_set = encoded_knn_copy.drop(
    columns=["price", "price_per_sqft", "street"]
)
for column in z_scaled_training_set.columns:
    z_scaled_training_set[column] = (
        z_scaled_training_set[column] - z_scaled_training_set[column].mean()
    ) / z_scaled_training_set[column].std()


features = z_scaled_training_set
# droping street as it is essentially a id, all are unique.
# price_per_sqft is just another copy of price, i can calculate it later if i need it
target = np.log1p(encoded_knn_copy.price)
price_bins = pd.qcut(target, q=10, labels=False, duplicates="drop")

training_set_features, testing_set_features, training_set_target, testing_set_target = (
    train_test_split(
        features, target, test_size=0.2, random_state=985, stratify=price_bins
    )  # 20% is the convention, might change as i do more testing
)

cv = ShuffleSplit(n_splits=30, test_size=0.2, random_state=30)
LR = LinearRegression()
cross_validation_lr = cross_validate(
    LR, training_set_features, training_set_target, cv=cv, scoring="r2"
)
print("mean:", cross_validation_lr["test_score"].mean())
print("std:", cross_validation_lr["test_score"].std())
print(
    "min/max:",
    cross_validation_lr["test_score"].min(),
    cross_validation_lr["test_score"].max(),
)

print("cross_val_lr: ", cross_validation_lr)
LR.fit(training_set_features, training_set_target)
KNN = KNeighborsRegressor()
cross_validation_training = cross_validate(
    KNN, training_set_features, training_set_target, cv=5
)
KNN.fit(training_set_features, training_set_target)

print(cross_validation_training)


KNN.fit(training_set_features, training_set_target)
final_score = KNN.score(testing_set_features, testing_set_target)
print(
    final_score
)  # score of 0.065 (atrocious, terrible and only a little better than guessing)


params = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 27],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}


# grid = GridSearchCV(KNN, params, cv=5, scoring="r2")
# grid.fit(training_set_features, training_set_target)
# print(grid.best_params_)
# print(grid.best_score_)


print("train:", LR.score(training_set_features, training_set_target))
print("test:", LR.score(testing_set_features, testing_set_target))

print("train price std:", training_set_target.std())
print("test price std:", testing_set_target.std())
