
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Load training and testing data
X_train, X_test, y_train, y_test = joblib.load(
    "models/train_test_data.pkl"
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "SVM": LinearSVC()
}

results = {}

# Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")

    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }

    print(f"\n{name} Results")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# Select best model based on F1 score
best_model_name = max(
    results,
    key=lambda name: results[name]["F1 Score"]
)

best_model = models[best_model_name]

# Save best model
joblib.dump(best_model, "models/sentiment_model.pkl")

print("\n------------------------------")
print("BEST MODEL:", best_model_name)
print("F1 SCORE:", results[best_model_name]["F1 Score"])
print("Model saved successfully!")
print("------------------------------")

# Save results
joblib.dump(results, "models/model_results.pkl")
