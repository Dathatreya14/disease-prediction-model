import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import RandomOverSampler

# -----------------------------
# 1. Load Dataset
# -----------------------------
file_path = "/content/disease prediction.xlsx"   # <-- change to your actual dataset file
data = pd.read_excel(file_path)

print("Columns in dataset:", data.columns.tolist())

# Features (X) and Target (y)
target_col = "disease"
X = data.drop(target_col, axis=1)
y = data[target_col]

# -----------------------------
# 2. Fix Class Imbalance (Oversampling)
# -----------------------------
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)

print("\nClass distribution after oversampling:")
print(y_resampled.value_counts())

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
)

# -----------------------------
# 4. Train Model
# -----------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# 5. Evaluation
# -----------------------------
y_pred = model.predict(X_test)
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -----------------------------
# 6. User Input Prediction
# -----------------------------
print("\nEnter patient details for disease prediction:")

user_data = {}
user_data["skin_rash"] = int(input("Skin Rash (0/1): "))
user_data["nodal_skin_eruptions"] = int(input("Nodal Skin Eruptions (0/1): "))
user_data["continuous_sneezing"] = int(input("Continuous Sneezing (0/1): "))
user_data["shivering"] = int(input("Shivering (0/1): "))
user_data["chills"] = int(input("Chills (0/1): "))
user_data["joint_pain"] = int(input("Joint Pain (0/1): "))
user_data["muscle_wasting"] = int(input("Muscle Wasting (0/1): "))
user_data["weight_gain"] = int(input("Weight Gain (0/1): "))
user_data["history_of_alcohol_consumption"] = int(input("History of Alcohol Consumption (0/1): "))
user_data["skin_peeling"] = int(input("Skin Peeling (0/1): "))

# Convert to DataFrame
user_df = pd.DataFrame([user_data])

# Prediction
prediction = model.predict(user_df)[0]
print("\nPredicted Disease:", prediction)
