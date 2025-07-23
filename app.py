import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import os
import hashlib
from streamlit_option_menu import option_menu
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

USERS_FILE = "users.csv"

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    if os.path.exists(USERS_FILE):
        return pd.read_csv(USERS_FILE).set_index("username").to_dict()["password"]
    return {}

def save_user(username, password):
    df = pd.DataFrame([[username, hash_password(password)]], columns=["username", "password"])
    if os.path.exists(USERS_FILE):
        df.to_csv(USERS_FILE, mode='a', header=False, index=False)
    else:
        df.to_csv(USERS_FILE, index=False)

users = load_users()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "current_user" not in st.session_state:
    st.session_state.current_user = ""
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "user_models" not in st.session_state:
    st.session_state.user_models = {}

st.set_page_config(page_title="Salary Prediction App", layout="wide")

if not st.session_state.authenticated:
    st.title("🔐 Login to Salary Prediction App")
    login_tab, signup_tab = st.tabs(["Login", "Signup"])

    with login_tab:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if username in users and users[username] == hash_password(password):
                st.session_state.authenticated = True
                st.session_state.current_user = username
                st.success("✅ Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials!")

    with signup_tab:
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        if st.button("Signup"):
            if new_username in users:
                st.warning("User already exists.")
            else:
                save_user(new_username, new_password)
                st.success("Account created! Please login.")
                users = load_users()
    st.stop()

def get_default_model():
    return joblib.load("best_model.pkl")

def get_neural_model():
    return joblib.load("mlp_model.pkl")

def load_model():
    return st.session_state.user_models.get(st.session_state.current_user, {}).get("model", get_default_model())

def load_scores():
    return pd.read_csv("model_scores.csv", index_col=0)

with st.sidebar:
    st.markdown(f"👤 Logged in as: `{st.session_state.current_user}`")
    theme = st.radio("🌗 Choose Theme", ["Light", "Dark"], index=0 if st.session_state.theme == "Light" else 1)
    st.session_state.theme = theme

    selected = option_menu(
        "Main Menu",
        ["Predict Salary", "Model Comparison", "Upload Custom Model", "Retrain Model"],
        icons=["bar-chart", "graph-up", "cloud-upload", "gear"],
        menu_icon="cast",
        default_index=0
    )

    if st.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.rerun()

if theme == "Dark":
    st.markdown("""
        <style>
            body { background-color: #1e1e1e; color: white; }
            .stButton>button { background-color: #333; color: white; }
        </style>
    """, unsafe_allow_html=True)

st.title("💼 Employeee Salary Prediction App")

if selected == "Predict Salary":
    st.header("📈 Enter Employee Information")

    model_choice = st.radio("Choose Model", ["Old Best Model", "Neural Network", "Custom/Uploaded"])
    if model_choice == "Old Best Model":
        model = get_default_model()
    elif model_choice == "Neural Network":
        model = get_neural_model()
    else:
        model = load_model()

    age = st.slider("Age", 18, 65, 30)
    workclass_options = ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked", "Other"]
    workclass = workclass_options.index(st.selectbox("Workclass", workclass_options))

    fnlwgt = st.number_input("fnlwgt", value=150000)
    educational_num = st.slider("Education Level (1=Preschool, 16=Doctorate)", 1, 16, 13)

    marital_status_options = ["Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Other"]
    marital_status = marital_status_options.index(st.selectbox("Marital Status", marital_status_options))

    occupation_options = ["Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"]
    occupation = occupation_options.index(st.selectbox("Occupation", occupation_options))

    relationship_options = ["Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]
    relationship = relationship_options.index(st.selectbox("Relationship", relationship_options))

    race_options = ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"]
    race = race_options.index(st.selectbox("Race", race_options))

    gender = 1 if st.radio("Gender", ["Female", "Male"]) == "Male" else 0
    capital_gain = st.number_input("Capital Gain", value=0)
    capital_loss = st.number_input("Capital Loss", value=0)
    hours_per_week = st.slider("Hours per Week", 1, 99, 40)

    country_options = ["United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]
    native_country = country_options.index(st.selectbox("Native Country", country_options))

    input_df = pd.DataFrame([{
        'age': age,
        'workclass': workclass,
        'fnlwgt': fnlwgt,
        'educational-num': educational_num,
        'marital-status': marital_status,
        'occupation': occupation,
        'relationship': relationship,
        'race': race,
        'gender': gender,
        'capital-gain': capital_gain,
        'capital-loss': capital_loss,
        'hours-per-week': hours_per_week,
        'native-country': native_country
    }])

    if st.button("🔍 Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][prediction] * 100
        label = ">50K" if prediction == 1 else "<=50K"
        st.success(f"Predicted Salary: {label}")
        st.info(f"Confidence: {proba:.2f}%")
        csv = input_df.copy()
        csv["Prediction"] = label
        st.download_button("📥 Download Prediction", csv.to_csv(index=False).encode(), "prediction.csv")

elif selected == "Model Comparison":
    st.header("📊 Model Accuracy Comparison")
    scores = load_scores()
    best_model = scores["Accuracy"].idxmax()

    scores_reset = scores.reset_index()
    scores_reset.columns = ["Model", "Accuracy"]
    fig = px.bar(
        scores_reset, x="Model", y="Accuracy", text="Accuracy",
        color="Model",
        color_discrete_sequence=["#4CAF50" if m == best_model else "#87CEEB" for m in scores.index]
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(title="Model Accuracy", yaxis_range=[0, 1])
    st.plotly_chart(fig)
    st.download_button("📥 Download Scores", scores.to_csv().encode(), "model_scores.csv")

elif selected == "Upload Custom Model":
    st.header("📤 Upload Your Trained Model (.pkl)")
    uploaded_model = st.file_uploader("Upload a .pkl file", type=["pkl"])
    if uploaded_model:
        model = joblib.load(uploaded_model)
        st.session_state.user_models.setdefault(st.session_state.current_user, {})["model"] = model
        st.success("✅ Model uploaded and saved to session!")

elif selected == "Retrain Model":
    st.header("🔁 Retrain Model from CSV")
    csv_file = st.file_uploader("Upload dataset (CSV)", type="csv")
    if csv_file:
        df = pd.read_csv(csv_file)
        try:
            df.drop(columns=["education"], inplace=True, errors='ignore')
            cat_cols = ['workclass', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income']
            encoder = LabelEncoder()
            for col in cat_cols:
                df[col] = encoder.fit_transform(df[col])
            X = df.drop(columns=['income'])
            y = df['income']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            pipe = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=1000))])
            pipe.fit(X_train, y_train)
            acc = accuracy_score(y_test, pipe.predict(X_test))
            st.session_state.user_models.setdefault(st.session_state.current_user, {})["model"] = pipe
            st.success(f"✅ Retrained model with accuracy: {acc:.2f}")
        except Exception as e:
            st.error(f"Error during retraining: {e}")