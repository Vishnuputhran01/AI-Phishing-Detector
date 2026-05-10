import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("phishing_dataset.csv")

# Features
X = data[[
    "has_https",
    "url_length",
    "special_chars",
    "suspicious_words",
    "ip_address"
]]

# Labels
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier()

model.fit(X_train, y_train)

# Predict test data
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)


# -------------------------------
# Feature Extractor Function
# -------------------------------
def extract_features(url):

    has_https = 1 if "https" in url else 0

    url_length = len(url)

    special_chars = (
        url.count("-") +
        url.count("@") +
        url.count("?") +
        url.count("=")
    )

    suspicious_words = 1 if any(word in url for word in [
        "login",
        "verify",
        "secure",
        "update",
        "bank",
        "account",
        "password",
        "signin"
    ]) else 0

    # Check if IP address exists
    domain_part = url.split("//")[-1].split("/")[0]

    ip_address = 1 if any(char.isdigit() for char in domain_part) else 0

    # Create DataFrame with column names
    features = pd.DataFrame([[
        has_https,
        url_length,
        special_chars,
        suspicious_words,
        ip_address
    ]], columns=[
        "has_https",
        "url_length",
        "special_chars",
        "suspicious_words",
        "ip_address"
    ])

    return features


# -------------------------------
# AI Explanation Function
# -------------------------------
def explain_prediction(url):

    reasons = []

    if "https" not in url:
        reasons.append("No HTTPS security")

    if len(url) > 30:
        reasons.append("Long suspicious URL")

    if any(word in url for word in [
        "login",
        "verify",
        "secure",
        "update",
        "bank",
        "account",
        "password"
    ]):
        reasons.append("Contains suspicious keywords")

    if "-" in url or "@" in url:
        reasons.append("Contains suspicious symbols")

    if any(char.isdigit() for char in url.split("//")[-1].split("/")[0]):
        reasons.append("Contains possible IP address")

    return reasons


# -------------------------------
# Test URL
# -------------------------------
url = "http://paypal-secure-login-update.com"

print("\nAnalyzing URL:")
print(url)

# Extract features
features = extract_features(url)

# Predict
result = model.predict(features)

# Probability
probabilities = model.predict_proba(features)

confidence = round(max(probabilities[0]) * 100, 2)

print("\nPrediction:", result[0])

print("Confidence:", confidence, "%")

# Explanation
reasons = explain_prediction(url)

print("\nReasons:")

for r in reasons:
    print("-", r)