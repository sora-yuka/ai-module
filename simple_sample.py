from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor


housing = datasets.fetch_california_housing(return_X_y=True, as_frame=True)

print(housing.feature_names)  # type: ignore

x = housing.data  # type: ignore
y = housing.target  # type: ignore

print(x[0])
print(y[0])

poly = PolynomialFeatures()
x = poly.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=432)


LR = LinearRegression()
GBR = GradientBoostingRegressor()
RFR = RandomForestRegressor()

for model in [LR, GBR, RFR]:
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    print(model, r2)
