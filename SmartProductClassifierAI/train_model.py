import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from numpy import unique
import joblib

# Load data
data = pd.read_csv("products.csv")

# Clean category column
data = data.dropna(subset=["category"])
data = data[data["category"].str.strip() != ""]

# Remove categories with < 2 occurrences
category_counts = data["category"].value_counts()
valid_categories = category_counts[category_counts >= 2].index
data = data[data["category"].isin(valid_categories)]

# Combine text fields
data["text"] = data["name"].fillna("") + " " + data["description"].fillna("")
X = data[["text", "price"]]
y = data["category"]

# Split BEFORE encoding
X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Encode only classes seen during training
le = LabelEncoder()
y_train = le.fit_transform(y_train_raw)

# Filter out test rows with unseen labels
X_test = X_test[y_test_raw.isin(le.classes_)]
y_test_raw = y_test_raw[y_test_raw.isin(le.classes_)]
y_test = le.transform(y_test_raw)

# Transformers
text_transformer = TfidfVectorizer(max_features=500)
price_transformer = StandardScaler()

preprocessor = ColumnTransformer(transformers=[
    ("text", text_transformer, "text"),
    ("price", price_transformer, ["price"])
])

# Model pipeline
model = Pipeline([
    ("features", preprocessor),
    ("classifier", XGBClassifier(
        eval_metric='mlogloss',
        n_estimators=150,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ))
])

# Train the model
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

# Align label names with only those present in y_test
present_labels = sorted(unique(y_test))
print("Classification Report:")
print(classification_report(
    y_test, y_pred,
    labels=present_labels,
    target_names=le.classes_[present_labels]
))

# Save model and label encoder
joblib.dump(model, "xgboost_classifier.pkl")
joblib.dump(le, "label_encoder.pkl")
print("✅ Model & label encoder saved.")
