from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score

df = pd.read_csv("./datasets/bedtime_screentime_sleep_debt.csv")

X = df.drop(columns=["user_id", "next_day_fatigue_score"])
y = df["next_day_fatigue_score"]

X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

LR = LinearRegression()
GBR = GradientBoostingRegressor()
RFR = RandomForestRegressor()

list_model = [LR, GBR, RFR]

for model in list_model:
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(model, r2)

new_person = pd.DataFrame([{
    "age": 30,
    "gender": "Female",
    "occupation_type": "Student",
    "chronotype": "Night Owl",
    "bedtime_phone_minutes": 20,
    "primary_bedtime_app": "TikTok / Reels",
    "screen_brightness_pct": 63,
    "blue_light_filter_active": 1,
    "caffeine_post_5pm_mg": 79,
    "physical_activity_min": 44,
    "sleep_latency_min": 65,
    "total_sleep_hours": 6.5,
    "deep_sleep_pct": 20,
    "rem_sleep_pct": 21,
    "morning_alarm_snoozes": 3,
}])

new_person = pd.get_dummies(new_person)
new_person = new_person.reindex(columns=X.columns, fill_value=0)


def predict(model: Any) -> Any:
    model.fit(X_train, y_train)
    prediction = model.predict(new_person)
    return round(prediction[0], 1)


print("Predicted fatigue score: ", predict(LR))
print("Predicted fatigue score: ", predict(GBR))
print("Predicted fatigue score: ", predict(RFR))

