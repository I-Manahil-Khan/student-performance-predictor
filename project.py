import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# PAGE CONFIG
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor System")
st.write("Machine Learning Based PASS / FAIL Prediction")
# DATABASE

conn = sqlite3.connect("students.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS student_predictions(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment1 REAL,
    assignment2 REAL,
    assignment3 REAL,
    quiz1 REAL,
    quiz2 REAL,
    midterm REAL,
    final_exam REAL,
    percentage REAL,
    result TEXT,
    suggestion TEXT
)
""")

conn.commit()

# LOAD DATASET
@st.cache_data
def load_data():
    df = pd.read_csv("student_data.csv")
    return df.head(300)

df = load_data()
# DATASET PREVIEW

st.subheader("📂 Dataset Preview")

st.dataframe(df)
# CREATE PASS/FAIL TARGET

df["Result"] = df["overall_score"].apply(
    lambda x: 1 if x >= 50 else 0
)
# FEATURE SELECTION

features = [
    "assignment1",
    "assignment2",
    "assignment3",
    "quiz1",
    "quiz2",
    "midterm",
    "final_exam"
]

X = df[features]
y = df["Result"]

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# MODELS

log_model = LogisticRegression(max_iter=1000)

tree_model = DecisionTreeClassifier(
    random_state=42
)

forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
# TRAINING

log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)
forest_model.fit(X_train, y_train)

# ACCURACY

log_acc = accuracy_score(
    y_test,
    log_model.predict(X_test)
)

tree_acc = accuracy_score(
    y_test,
    tree_model.predict(X_test)
)

forest_acc = accuracy_score(
    y_test,
    forest_model.predict(X_test)
)
# MODEL COMPARISON

st.subheader("📊 Model Accuracy Comparison")

accuracy_df = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest"
    ],
    "Accuracy (%)": [
        round(log_acc * 100, 2),
        round(tree_acc * 100, 2),
        round(forest_acc * 100, 2)
    ]
})

st.dataframe(accuracy_df)

# BEST MODEL

best_accuracy = max(
    log_acc,
    tree_acc,
    forest_acc
)

if best_accuracy == log_acc:
    best_model = log_model
    best_name = "Logistic Regression"

elif best_accuracy == tree_acc:
    best_model = tree_model
    best_name = "Decision Tree"

else:
    best_model = forest_model
    best_name = "Random Forest"

st.success(f"✅ Best Model: {best_name}")

# PASS FAIL CHART

st.subheader("📈 Pass / Fail Distribution")

pass_count = (df["Result"] == 1).sum()
fail_count = (df["Result"] == 0).sum()

fig, ax = plt.subplots()

ax.bar(
    ["Pass", "Fail"],
    [pass_count, fail_count]
)

ax.set_ylabel("Number of Students")

st.pyplot(fig)

# NEW STUDENT DATA ENTRY

st.subheader("📝 Enter Student Marks")

col1, col2 = st.columns(2)

with col1:

    assignment1 = st.number_input(
        "Assignment 1",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    assignment2 = st.number_input(
        "Assignment 2",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    assignment3 = st.number_input(
        "Assignment 3",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    quiz1 = st.number_input(
        "Quiz 1",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

with col2:

    quiz2 = st.number_input(
        "Quiz 2",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    midterm = st.number_input(
        "Midterm",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    final_exam = st.number_input(
        "Final Exam",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )
# PREDICTION

if st.button("Predict Student Result"):

    new_student = pd.DataFrame([[
        assignment1,
        assignment2,
        assignment3,
        quiz1,
        quiz2,
        midterm,
        final_exam
    ]], columns=features)

    prediction = best_model.predict(new_student)

    percentage = (
        assignment1 +
        assignment2 +
        assignment3 +
        quiz1 +
        quiz2 +
        midterm +
        final_exam
    ) / 7

    st.subheader("📊 Student Report")

    st.write(f"Overall Percentage: {percentage:.2f}%")

    result = "PASS" if percentage >= 50 else "FAIL"

    if result == "PASS":
        st.success("🎉 PASS")
    else:
        st.error("❌ FAIL")

    if hasattr(best_model, "predict_proba"):

        prob = best_model.predict_proba(
            new_student
        )

        st.write(
            f"✅ Pass Probability: {prob[0][1]*100:.2f}%"
        )

        st.write(
            f"❌ Fail Probability: {prob[0][0]*100:.2f}%"
        )

    st.subheader("💡 Suggestion")

    if percentage <= 40:

        suggestion = "Critical Risk"

        st.error("""
Your performance is extremely low.

• Attend classes regularly
• Complete all assignments
• Seek academic support immediately
""")

    elif percentage <= 59:

        suggestion = "At Risk"

        st.warning("""
You are below passing level.

• Focus on quizzes
• Improve exam preparation
• Increase study hours
""")

    elif percentage <= 69:

        suggestion = "Average"

        st.info("""
You have passed.

• Practice more
• Improve weak subjects
• Aim for higher grades
""")

    elif percentage <= 84:

        suggestion = "Good"

        st.success("""
Good Performance.

• Maintain consistency
• Keep working hard
• Continue regular studies
""")

    else:

        suggestion = "Excellent"

        st.success("""
Excellent Performance.

• Outstanding work
• Maintain your hard work
• Keep achieving high scores
""")

    # SAVE TO DATABASE
    cursor.execute("""
    INSERT INTO student_predictions(
        assignment1,
        assignment2,
        assignment3,
        quiz1,
        quiz2,
        midterm,
        final_exam,
        percentage,
        result,
        suggestion
    )
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """,
    (
        assignment1,
        assignment2,
        assignment3,
        quiz1,
        quiz2,
        midterm,
        final_exam,
        percentage,
        result,
        suggestion
    ))

    conn.commit()

    st.success("✅ Student Record Saved Successfully")

# SAVED PREDICTIONS

st.subheader("💾 Saved Student Predictions")

saved_df = pd.read_sql(
    "SELECT * FROM student_predictions",
    conn
)

st.dataframe(saved_df)

# AT RISK STUDENTS

st.subheader("🚨 Students Below 50%")

risk_students = df[
    df["overall_score"] < 60
]

st.dataframe(risk_students)

# PROJECT INFO

st.subheader("ℹ️ About Project")

st.write("""
This system predicts student PASS/FAIL status using Machine Learning.

Models Used:
• Logistic Regression
• Decision Tree
• Random Forest

Input Features:
• Assignment Marks
• Quiz Marks
• Midterm Marks
• Final Exam Marks

Output:
• PASS / FAIL
• Percentage
• Probability
• Personalized Suggestion
• Database Storage
""")