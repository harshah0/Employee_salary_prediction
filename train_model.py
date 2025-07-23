import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load and preprocess data
df = pd.read_csv("adult.csv")
df.drop(columns=["education"], inplace=True, errors='ignore')

cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

X = df.drop(columns=['income'])
y = df['income']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Neural Network": MLPClassifier(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(probability=True)
}

scores = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipe.fit(X_train, y_train)
    acc = accuracy_score(y_test, pipe.predict(X_test))
    scores[name] = acc

    if name == "Logistic Regression":
        joblib.dump(pipe, "best_model.pkl")
    elif name == "Neural Network":
        joblib.dump(pipe, "mlp_model.pkl")

# Save model scores
scores_df = pd.DataFrame.from_dict(scores, orient="index", columns=["Accuracy"])
scores_df.to_csv("model_scores.csv")
print("✅ Training complete. Models and scores saved.")
