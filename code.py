
# ---------- INSTALL LIBRARIES ----------
!pip install requests pandas scikit-learn imbalanced-learn tldextract

# ---------- IMPORT LIBRARIES ----------
import requests
import pandas as pd
import re
import socket
import tldextract
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# ===============================
# STEP 1: FETCH LIVE PHISHING DATA
# ===============================

def fetch_live_phishing(limit=500):
    url = "https://openphish.com/feed.txt"
    response = requests.get(url, timeout=10)
    urls = response.text.split("\n")[:limit]

    phishing_data = [
        {"url": u.strip(), "label": 1}
        for u in urls if u.strip()
    ]

    return pd.DataFrame(phishing_data)

phishing_df = fetch_live_phishing()
print("Phishing URLs Collected:", len(phishing_df))


# ===============================
# STEP 2: FETCH LIVE LEGITIMATE DATA
# ===============================

def fetch_live_legit():
    legit_domains = [
        "google.com","wikipedia.org","github.com",
        "microsoft.com","apple.com","amazon.com",
        "linkedin.com","youtube.com","cloudflare.com",
        "stackoverflow.com","reddit.com"
    ]

    legit_data = []

    for domain in legit_domains:
        try:
            socket.gethostbyname(domain)
            legit_data.append({
                "url": "http://" + domain,
                "label": 0
            })
        except:
            pass

    return pd.DataFrame(legit_data)

legit_df = fetch_live_legit()
print("Legitimate URLs Collected:", len(legit_df))


# ===============================
# STEP 3: COMBINE DATA
# ===============================

df = pd.concat([phishing_df, legit_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)

print("Total Dataset Size:", len(df))


# ===============================
# STEP 4: FEATURE EXTRACTION
# ===============================

def extract_features(url):
    features = {}

    features["url_length"] = len(url)
    features["num_dots"] = url.count(".")
    features["num_digits"] = sum(c.isdigit() for c in url)
    features["num_hyphens"] = url.count("-")
    features["has_ip"] = 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0
    features["has_https"] = 1 if url.startswith("https") else 0

    ext = tldextract.extract(url)
    domain = ext.domain + "." + ext.suffix

    try:
        socket.gethostbyname(domain)
        features["dns_resolves"] = 1
    except:
        features["dns_resolves"] = 0

    return features

print("Extracting features...")

feature_df = df["url"].apply(lambda x: extract_features(str(x)))
feature_df = pd.json_normalize(feature_df)

final_dataset = pd.concat([feature_df, df["label"]], axis=1)

print("Feature Extraction Completed")


# ===============================
# STEP 5: SAVE DATASET
# ===============================

file_name = "live_dataset_exp3.csv"
final_dataset.to_csv(file_name, index=False)
print("Dataset saved as:", file_name)


# ===============================
# STEP 6: HANDLE CLASS IMBALANCE
# ===============================

X = final_dataset.drop("label", axis=1)
y = final_dataset["label"]

smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE Balancing:", len(X_resampled))


# ===============================
# STEP 7: TRAIN-TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


# ===============================
# STEP 8: TRAIN RANDOM FOREST MODEL
# ===============================

model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

predictions = model.predict(X_test)


# ===============================
# STEP 9: EVALUATION
# ===============================

print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, predictions))


# ===============================
# STEP 10: FEATURE IMPORTANCE
# ===============================

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importance:\n")
print(importance.sort_values(ascending=False))


# ===============================
# STEP 11: CUSTOM URL TESTING
# ===============================

def predict_url(url):
    features = extract_features(url)
    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    if prediction == 1:
        return "⚠️ Phishing Website"
    else:
        return "✅ Legitimate Website"

print("\nTest Example:")
print(predict_url("http://secure-paypal-update-login.com"))
