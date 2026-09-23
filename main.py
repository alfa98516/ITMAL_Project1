import kagglehub
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

path = kagglehub.dataset_download("debayank2024/house-price-prediction")

knn_copy = pd.read_csv(path + "/modified_data.csv")


# convert date to integer values so i can properly work with it
# not using year as the data is all within the same year
knn_copy["date"] = pd.to_datetime(knn_copy["date"])
knn_copy["month"] = knn_copy["date"].dt.month
knn_copy["day"] = knn_copy["date"].dt.day
knn_copy = knn_copy.drop(columns=["date"])

print(len(knn_copy.columns))

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


target = encoded_knn_copy.price
features = z_scaled_training_set
# droping street as it is essentially a id, all are unique.
# price_per_sqft is just another copy of price, i can calculate it later if i need it


training_set_features, testing_set_features, training_set_target, testing_set_target = (
    train_test_split(
        features, target, test_size=0.2, random_state=10
    )  # 20% is the convention, might change as i do more testing
)

LR = LinearRegression()
cross_validation_lr = cross_validate(
    LR, training_set_features, training_set_target, cv=5
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
