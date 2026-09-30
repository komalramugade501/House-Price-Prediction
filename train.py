from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"MAE: {mean_absolute_error(y_test, predictions):.4f}")
print(f"R² Score: {r2_score(y_test, predictions):.4f}")

importance = dict(sorted(
    zip(X.columns, model.feature_importances_),
    key=lambda item: item[1],
    reverse=True
))
print("\nFeature Importance:")
for feature, score in importance.items():
    print(f"{feature}: {score:.4f}")
