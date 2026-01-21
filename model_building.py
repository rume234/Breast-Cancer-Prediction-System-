import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_breast_cancer

# 1. Load Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['diagnosis'] = data.target # 0: Malignant, 1: Benign

# 2. Feature Selection (Choosing 5 specific features)
# Mapping standard names to the specific list provided in your prompt
selected_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
X = df[selected_features]
y = df['diagnosis']

# 3. Preprocessing
# Scale features (Mandatory for Logistic Regression)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 4. Implement Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)

# 5. Evaluate
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))

# 6. Save Model and Scaler
# We save the scaler too because new input must be scaled the same way!
joblib.dump(model, 'model/breast_cancer_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("Model saved successfully.")