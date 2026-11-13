import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from auth import init_db, register_user, login_user
from ui_components import apply_custom_css, render_header, render_metric_card, render_risk_banner

# ---------------------------------------------------------
# 1. Page Configuration & Initialization
# ---------------------------------------------------------
st.set_page_config(
    page_title="Academic AI & XAI Performance Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_custom_css()
init_db()

# Course Catalog (University Curriculum Context)
COURSE_LIST = [
    "CSE-101: Structured Programming",
    "CSE-203: Data Structures & Algorithms",
    "CSE-305: Artificial Intelligence & Machine Learning",
    "CSE-311: Database Management Systems",
    "EEE-101: Electrical Circuits & Electronics",
    "MAT-101: Differential & Integral Calculus",
    "ENG-102: Academic Writing & Communication"
]

# Initialize History Table with Course Column
def init_history_table():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            student_name TEXT,
            course TEXT,
            attendance REAL,
            prev_cgpa REAL,
            predicted_cgpa REAL,
            risk_level TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Ensure 'course' column exists if upgrading from an older DB
    c.execute("PRAGMA table_info(prediction_history)")
    columns = [col[1] for col in c.fetchall()]
    if "course" not in columns:
        c.execute("ALTER TABLE prediction_history ADD COLUMN course TEXT DEFAULT 'General Semester Evaluation'")
    conn.commit()
    conn.close()

init_history_table()

def save_prediction_record(username, student_name, course, attendance, prev_cgpa, pred_cgpa, risk_label):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO prediction_history (username, student_name, course, attendance, prev_cgpa, predicted_cgpa, risk_level)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (username, student_name, course, attendance, prev_cgpa, pred_cgpa, risk_label))
    conn.commit()
    conn.close()

# ---------------------------------------------------------
# 2. Load Models, Scaler, Dataset & Metrics
# ---------------------------------------------------------
@st.cache_resource
def load_artifacts():
    cgpa_model = joblib.load("cgpa_model.pkl")
    risk_model = joblib.load("risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    df = pd.read_csv("student_data.csv")
    return cgpa_model, risk_model, scaler, metrics, df

try:
    cgpa_model, risk_model, scaler, model_metrics, dataset_df = load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}. Please run `python train_model.py` first.")
    st.stop()

# ---------------------------------------------------------
# 3. Prediction & Explainable AI (XAI) Engine
# ---------------------------------------------------------
FEATURE_COLS = [
    "attendance", "prev_cgpa", "study_hours", "social_media_hours",
    "tuition", "quiz_avg", "mid_score", "teacher_rating", "study_balance"
]

def predict_and_explain(attendance, prev_cgpa, study_hours, social_media_hours, tuition, quiz_avg, mid_score, teacher_rating):
    study_balance = round(study_hours - (social_media_hours * 0.4), 2)
    input_dict = {
        "attendance": attendance,
        "prev_cgpa": prev_cgpa,
        "study_hours": study_hours,
        "social_media_hours": social_media_hours,
        "tuition": tuition,
        "quiz_avg": quiz_avg,
        "mid_score": mid_score,
        "teacher_rating": teacher_rating,
        "study_balance": study_balance
    }
    input_df = pd.DataFrame([input_dict])[FEATURE_COLS]
    scaled_input = scaler.transform(input_df)

    pred_cgpa = float(np.clip(cgpa_model.predict(scaled_input)[0], 2.00, 4.00))
    risk_code = int(risk_model.predict(scaled_input)[0])
    risk_probs = risk_model.predict_proba(scaled_input)[0]

    # Override safeguard for extreme critical academic thresholds
    if pred_cgpa < 2.75 or attendance < 60.0:
        risk_code = 2
    elif (pred_cgpa < 3.25 or attendance < 75.0) and risk_code == 0:
        risk_code = 1

    risk_map = {0: "Safe", 1: "Moderate Risk", 2: "High Risk"}
    risk_label = risk_map[risk_code]

    # Local XAI Feature Contribution Calculation (Sensitivity Perturbation)
    baseline_vals = {
        "attendance": 78.0,
        "prev_cgpa": 3.15,
        "study_hours": 3.2,
        "social_media_hours": 3.3,
        "tuition": 0,
        "quiz_avg": 9.5,
        "mid_score": 15.5,
        "teacher_rating": 3.0,
        "study_balance": 1.88
    }

    contributions = {}
    for feat in FEATURE_COLS:
        perturbed_dict = input_dict.copy()
        perturbed_dict[feat] = baseline_vals[feat]
        if feat in ["study_hours", "social_media_hours"]:
            perturbed_dict["study_balance"] = round(
                perturbed_dict["study_hours"] - (perturbed_dict["social_media_hours"] * 0.4), 2
            )
        p_df = pd.DataFrame([perturbed_dict])[FEATURE_COLS]
        p_scaled = scaler.transform(p_df)
        p_cgpa = float(np.clip(cgpa_model.predict(p_scaled)[0], 2.00, 4.00))
        contributions[feat] = round(pred_cgpa - p_cgpa, 4)

    return pred_cgpa, risk_code, risk_label, risk_probs, contributions, study_balance

def generate_counseling_plan(attendance, study_hours, social_media_hours, quiz_avg, mid_score, pred_cgpa, course):
    advice = []
    if attendance < 75.0:
        advice.append(f"🚨 **Attendance Recovery ({course}):** Your attendance is `{attendance}%`, which is below the 75% university benchmark. Attend the next 10 consecutive lectures to avoid exam debarment.")
    if study_hours < 3.0:
        advice.append(f"📖 **Structured Self-Study:** Increase daily study time from `{study_hours} hrs` to at least `3.5–4.0 hrs` focusing on {course} problem sets.")
    if social_media_hours > 3.5:
        advice.append(f"📵 **Digital Distraction Control:** Reduce social media usage from `{social_media_hours} hrs/day` to under `2.0 hrs/day` using Pomodoro focus blocks.")
    if quiz_avg < 9.0:
        advice.append(f"✍️ **Class Test (CT) Improvement:** Your CT average (`{quiz_avg}/15`) needs attention. Review weekly lecture slides before every quiz.")
    if mid_score < 14.0:
        advice.append(f"🎯 **Midterm Recovery Strategy:** With a Midterm score of `{mid_score}/25`, schedule office hours with your {course} instructor to cover weak topics before finals.")
    if not advice:
        advice.append(f"🌟 **Strong Academic Trajectory in {course}:** Maintain your current study habits and aim for peer-mentoring or research projects to sustain a `{pred_cgpa:.2f}` CGPA.")
    return advice

# ---------------------------------------------------------
# 4. Session State & Authentication Portal (No Auto-Login)
# ---------------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "username" not in st.session_state:
    st.session_state["username"] = ""
if "role" not in st.session_state:
    st.session_state["role"] = ""

if not st.session_state["logged_in"]:
    render_header(
        "🎓 Academic AI & Explainable AI (XAI) Portal",
        "Early Risk Identification, CGPA Forecasting & Personalized Student Counseling System"
    )

    col_left, col_center, col_right = st.columns([1, 1.4, 1])
    with col_center:
        auth_tab1, auth_tab2 = st.tabs(["🔐 Login to Portal", "📝 Create New Account"])

        with auth_tab1:
            st.subheader("Sign In")
            login_user_input = st.text_input("Username", key="login_username", placeholder="Enter your username")
            login_pass_input = st.text_input("Password", type="password", key="login_password", placeholder="Enter your password")
            if st.button("Login", use_container_width=True, type="primary"):
                if login_user_input and login_pass_input:
                    success, role = login_user(login_user_input.strip(), login_pass_input)
                    if success:
                        st.session_state["logged_in"] = True
                        st.session_state["username"] = login_user_input.strip()
                        st.session_state["role"] = role
                        st.rerun()
                    else:
                        st.error("Invalid username or password. Please try again.")
                else:
                    st.warning("Please enter both username and password.")

        with auth_tab2:
            st.subheader("Register New User")
            reg_user_input = st.text_input("Choose Username", key="reg_username", placeholder="e.g., sumaiya_cse")
            reg_pass_input = st.text_input("Create Password", type="password", key="reg_password", placeholder="Minimum 4 characters")
            reg_role_input = st.selectbox("Select Role", ["Student", "Teacher"], key="reg_role")
            if st.button("Create Account", use_container_width=True):
                if len(reg_user_input.strip()) < 3 or len(reg_pass_input) < 4:
                    st.warning("Username must be at least 3 characters and password at least 4 characters.")
                else:
                    created = register_user(reg_user_input.strip(), reg_pass_input, reg_role_input)
                    if created:
                        st.success("✅ Account created successfully! Please switch to the **'🔐 Login to Portal'** tab above to sign in.")
                    else:
                        st.error("Username already exists. Please choose a different username.")

        with st.expander("ℹ️ Quick Demo Credentials"):
            st.markdown("""
            - **Teacher Account:** Username: `teacher` | Password: `password123` (or register a new Teacher account)
            - **Student Account:** Register a new Student account and log in via the Login tab.
            """)
    st.stop()

# ---------------------------------------------------------
# 5. Sidebar Navigation (Logged-In State)
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 👤 {st.session_state['username']}")
    st.caption(f"Role: **{st.session_state['role']} Portal**")
    st.divider()

    if st.session_state["role"] == "Student":
        menu = st.radio(
            "Navigation",
            [
                "🎯 Course Performance & XAI",
                "🎛️ 'What-If' Habit Simulator",
                "📜 My Evaluation History",
                "🎓 5-Pillar Thesis Overview"
            ]
        )
    else:
        menu = st.radio(
            "Teacher Navigation",
            [
                "🧑‍🏫 Single Student & Course Evaluation",
                "📂 Batch Class Prediction (CSV)",
                "📊 EDA & Model Comparison",
                "📜 All Student Records",
                "🎓 5-Pillar Thesis Overview"
            ]
        )

    st.divider()
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state["logged_in"] = False
        st.session_state["username"] = ""
        st.session_state["role"] = ""
        st.rerun()

# ---------------------------------------------------------
# 6. STUDENT PORTAL VIEWS
# ---------------------------------------------------------
if st.session_state["role"] == "Student":
    if menu == "🎯 Course Performance & XAI":
        render_header(
            "🎯 Student Course Performance & XAI Predictor",
            "Select your course and enter academic metrics to forecast CGPA and understand key drivers."
        )

        with st.container():
            col_c1, col_c2 = st.columns([1.5, 1])
            with col_c1:
                selected_course = st.selectbox("📚 Select Course", COURSE_LIST)
            with col_c2:
                student_display_name = st.text_input("Student Name / ID", value=st.session_state["username"])

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                attendance = st.slider("Attendance (%)", 45.0, 100.0, 80.0, 0.5)
                prev_cgpa = st.number_input("Previous CGPA (4.00)", 2.00, 4.00, 3.20, 0.01)
            with c2:
                study_hours = st.slider("Daily Study Hours", 0.5, 8.5, 3.5, 0.1)
                social_media = st.slider("Social Media (hrs/day)", 0.5, 7.0, 2.5, 0.1)
            with c3:
                quiz_avg = st.slider("Class Test / Quiz Avg (/15)", 2.0, 15.0, 10.5, 0.5)
                mid_score = st.slider("Midterm Score (/25)", 4.0, 25.0, 17.0, 0.5)
            with c4:
                tuition_opt = st.selectbox("Extra Coaching / Tuition", ["No", "Yes"])
                tuition_val = 1 if tuition_opt == "Yes" else 0
                teacher_rating = st.slider("Class Engagement Rating (1-5)", 1, 5, 4)

        if st.button("🔍 Predict Performance & Generate XAI Report", type="primary", use_container_width=True):
            pred_cgpa, risk_code, risk_label, risk_probs, contributions, study_balance = predict_and_explain(
                attendance, prev_cgpa, study_hours, social_media, tuition_val, quiz_avg, mid_score, teacher_rating
            )
            save_prediction_record(
                st.session_state["username"], student_display_name, selected_course,
                attendance, prev_cgpa, pred_cgpa, risk_label
            )

            st.info(f"📚 **Evaluated Course:** {selected_course} | **Student:** {student_display_name}")
            render_risk_banner(risk_code, risk_label)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Predicted Final CGPA", f"{pred_cgpa:.2f} / 4.00", f"{pred_cgpa - prev_cgpa:+.2f} vs Prev")
            with m2:
                render_metric_card("Risk Classification", risk_label, f"Confidence: {max(risk_probs)*100:.1f}%")
            with m3:
                render_metric_card("Study Balance Index", f"{study_balance:.2f}", "Study - 0.4×Social")
            with m4:
                render_metric_card("Course Attendance", f"{attendance:.1f}%", "Min Req: 75.0%")

            st.subheader("🧠 Explainable AI (XAI): Why did the model predict this?")
            xai_df = pd.DataFrame({
                "Feature": list(contributions.keys()),
                "CGPA Impact": list(contributions.values())
            }).sort_values(by="CGPA Impact", ascending=True)
            xai_df["Direction"] = np.where(xai_df["CGPA Impact"] >= 0, "Positive Driver (+)", "Negative Drag (-)")

            fig_xai = px.bar(
                xai_df, x="CGPA Impact", y="Feature", color="Direction", orientation="h",
                color_discrete_map={"Positive Driver (+)": "#16a34a", "Negative Drag (-)": "#dc2626"},
                title=f"Local Feature Attribution for {selected_course}"
            )
            fig_xai.update_layout(height=380)
            st.plotly_chart(fig_xai, use_container_width=True)

            st.subheader("📋 Personalized Academic Counseling Plan")
            for item in generate_counseling_plan(attendance, study_hours, social_media, quiz_avg, mid_score, pred_cgpa, selected_course):
                st.markdown(f"- {item}")

    elif menu == "🎛️ 'What-If' Habit Simulator":
        render_header(
            "🎛️ Interactive 'What-If' Habit & Score Simulator",
            "Simulate how improving your attendance, study hours, or exam scores changes your expected CGPA in a course."
        )
        sim_course = st.selectbox("📚 Select Course to Simulate", COURSE_LIST)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📌 Current Baseline")
            b_att = st.number_input("Current Attendance (%)", 45.0, 100.0, 68.0)
            b_prev = st.number_input("Previous CGPA", 2.00, 4.00, 2.95)
            b_study = st.number_input("Current Study Hours/Day", 0.5, 8.5, 2.0)
            b_social = st.number_input("Current Social Media Hours/Day", 0.5, 7.0, 4.5)
            b_quiz = st.number_input("Current Quiz Avg (/15)", 2.0, 15.0, 8.5)
            b_mid = st.number_input("Current Midterm (/25)", 4.0, 25.0, 13.5)

        with col2:
            st.markdown("#### 🚀 Target 'What-If' Adjustments")
            t_att = st.slider("Target Attendance (%)", 45.0, 100.0, 88.0)
            t_study = st.slider("Target Study Hours/Day", 0.5, 8.5, 4.5)
            t_social = st.slider("Target Social Media Hours/Day", 0.5, 7.0, 1.5)
            t_quiz = st.slider("Target Quiz Avg (/15)", 2.0, 15.0, 12.5)
            t_mid = st.slider("Target Midterm (/25)", 4.0, 25.0, 19.0)

        base_cgpa, _, base_risk, _, _, _ = predict_and_explain(b_att, b_prev, b_study, b_social, 0, b_quiz, b_mid, 3)
        new_cgpa, _, new_risk, _, _, _ = predict_and_explain(t_att, b_prev, t_study, t_social, 0, t_quiz, t_mid, 4)

        st.divider()
        st.markdown(f"### 📊 Simulation Outcome for **{sim_course}**")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            render_metric_card("Baseline Predicted CGPA", f"{base_cgpa:.2f}", f"Status: {base_risk}")
        with sc2:
            render_metric_card("Simulated New CGPA", f"{new_cgpa:.2f}", f"{new_cgpa - base_cgpa:+.2f} Gain")
        with sc3:
            render_metric_card("Projected Risk Status", new_risk, f"Was: {base_risk}")

    elif menu == "📜 My Evaluation History":
        render_header("📜 My Course Evaluation History", "Review your past predictions across different courses.")
        conn = sqlite3.connect("users.db")
        hist_df = pd.read_sql_query(
            "SELECT student_name, course, attendance, prev_cgpa, predicted_cgpa, risk_level, timestamp FROM prediction_history WHERE username = ? ORDER BY id DESC",
            conn, params=(st.session_state["username"],)
        )
        conn.close()
        if hist_df.empty:
            st.info("No evaluations saved yet. Go to 'Course Performance & XAI' to run your first prediction.")
        else:
            st.dataframe(hist_df, use_container_width=True)

    elif menu == "🎓 5-Pillar Thesis Overview":
        render_header("🎓 Academic Thesis Architecture (5 Pillars)", "Methodology, Dataset, Preprocessing, Modeling & XAI Contribution")
        st.markdown("""
        1. **Problem Solving:** Early detection of academically at-risk students before final exams using multi-dimensional academic and behavioral indicators.
        2. **Dataset Design:** 800 university student records calibrated to the 4.00 CGPA scale, CT (/15), and Midterm (/25) assessments.
        3. **EDA & Preprocessing:** Missing value checks, outlier clipping, `StandardScaler` normalization, and engineered `study_balance` feature.
        4. **Modeling & Evaluation:** Comparative training across **Random Forest**, **XGBoost**, and **SVM** classifiers/regressors evaluated via Accuracy, F1-Score, and R².
        5. **Explainable AI (XAI) Contribution:** Local feature attribution explaining *why* a student received a specific prediction alongside actionable counseling plans.
        """)

# ---------------------------------------------------------
# 7. TEACHER / ADMIN PORTAL VIEWS
# ---------------------------------------------------------
else:
    if menu == "🧑‍🏫 Single Student & Course Evaluation":
        render_header(
            "🧑‍🏫 Instructor Student & Course Evaluation",
            "Evaluate individual student performance by course with teacher behavioral ratings and XAI diagnostics."
        )

        col_t1, col_t2 = st.columns([1.5, 1])
        with col_t1:
            selected_course = st.selectbox("📚 Select Course", COURSE_LIST)
        with col_t2:
            student_name = st.text_input("Student Name / Roll ID", value="Student_01")

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            attendance = st.slider("Attendance (%)", 45.0, 100.0, 72.0, 0.5)
            prev_cgpa = st.number_input("Previous CGPA", 2.00, 4.00, 2.85, 0.01)
        with c2:
            study_hours = st.slider("Study Hours/Day", 0.5, 8.5, 2.4, 0.1)
            social_media = st.slider("Social Media Hours/Day", 0.5, 7.0, 3.8, 0.1)
        with c3:
            quiz_avg = st.slider("Class Test / Quiz Avg (/15)", 2.0, 15.0, 8.5, 0.5)
            mid_score = st.slider("Midterm Score (/25)", 4.0, 25.0, 13.0, 0.5)
        with c4:
            tuition_val = 1 if st.selectbox("Private Tuition", ["No", "Yes"]) == "Yes" else 0
            teacher_rating = st.slider("Instructor Behavioral Rating (1-5)", 1, 5, 3)

        if st.button("🚀 Run Instructor Assessment", type="primary", use_container_width=True):
            pred_cgpa, risk_code, risk_label, risk_probs, contributions, study_balance = predict_and_explain(
                attendance, prev_cgpa, study_hours, social_media, tuition_val, quiz_avg, mid_score, teacher_rating
            )
            save_prediction_record(
                st.session_state["username"], student_name, selected_course,
                attendance, prev_cgpa, pred_cgpa, risk_label
            )

            st.info(f"📚 **Course:** {selected_course} | **Student:** {student_name}")
            render_risk_banner(risk_code, risk_label)

            m1, m2, m3 = st.columns(3)
            with m1:
                render_metric_card("Predicted Course/Final CGPA", f"{pred_cgpa:.2f}", f"Prev: {prev_cgpa:.2f}")
            with m2:
                render_metric_card("Risk Status", risk_label, f"Prob: {max(risk_probs)*100:.1f}%")
            with m3:
                render_metric_card("Instructor Rating", f"{teacher_rating} / 5", f"Study Balance: {study_balance:.2f}")

            xai_df = pd.DataFrame({
                "Feature": list(contributions.keys()),
                "Impact": list(contributions.values())
            }).sort_values(by="Impact", ascending=True)
            fig = px.bar(
                xai_df, x="Impact", y="Feature", orientation="h",
                color=xai_df["Impact"] >= 0,
                color_discrete_map={True: "#16a34a", False: "#dc2626"},
                title=f"XAI Risk & CGPA Drivers ({student_name} - {selected_course})"
            )
            fig.update_layout(showlegend=False, height=360)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("🩺 Recommended Faculty Intervention")
            for item in generate_counseling_plan(attendance, study_hours, social_media, quiz_avg, mid_score, pred_cgpa, selected_course):
                st.markdown(f"- {item}")

    elif menu == "📂 Batch Class Prediction (CSV)":
        render_header("📂 Batch Course / Class Prediction", "Upload a class CSV sheet to identify all at-risk students at once.")
        batch_course = st.selectbox("📚 Assign Course to Batch Sheet", COURSE_LIST)
        uploaded_file = st.file_uploader("Upload Student CSV", type=["csv"])

        sample_btn = st.checkbox("Use built-in sample batch (first 25 students from dataset)")
        if uploaded_file is not None or sample_btn:
            batch_df = pd.read_csv(uploaded_file) if uploaded_file else dataset_df.head(25).copy()
            req_cols = ["attendance", "prev_cgpa", "study_hours", "social_media_hours", "tuition", "quiz_avg", "mid_score", "teacher_rating"]
            if all(c in batch_df.columns for c in req_cols):
                batch_df["study_balance"] = np.round(batch_df["study_hours"] - (batch_df["social_media_hours"] * 0.4), 2)
                X_batch = scaler.transform(batch_df[FEATURE_COLS])
                batch_df["Course"] = batch_course
                batch_df["Predicted_CGPA"] = np.round(np.clip(cgpa_model.predict(X_batch), 2.00, 4.00), 2)
                risk_preds = risk_model.predict(X_batch)
                risk_map = {0: "Safe", 1: "Moderate Risk", 2: "High Risk"}
                batch_df["Risk_Status"] = [risk_map[int(r)] for r in risk_preds]

                st.success(f"Processed {len(batch_df)} student records for **{batch_course}**.")
                st.dataframe(batch_df[["Course", "attendance", "prev_cgpa", "quiz_avg", "mid_score", "Predicted_CGPA", "Risk_Status"]], use_container_width=True)

                at_risk_df = batch_df[batch_df["Risk_Status"] != "Safe"]
                st.download_button(
                    "⬇️ Download Filtered At-Risk Student List (CSV)",
                    data=at_risk_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"at_risk_{batch_course.split(':')[0]}.csv",
                    mime="text/csv"
                )
            else:
                st.error(f"CSV must contain columns: {req_cols}")

    elif menu == "📊 EDA & Model Comparison":
        render_header("📊 Exploratory Data Analysis (EDA) & Model Benchmark", "Dataset insights and comparative ML evaluation metrics.")
        tab_eda, tab_ml = st.tabs(["📈 Dataset EDA", "🤖 ML Model Comparison"])

        with tab_eda:
            c1, c2 = st.columns(2)
            with c1:
                fig_hist = px.histogram(dataset_df, x="predicted_cgpa", nbins=25, title="Distribution of Predicted CGPA", color_discrete_sequence=["#1E3A5F"])
                st.plotly_chart(fig_hist, use_container_width=True)
            with c2:
                fig_scat = px.scatter(dataset_df, x="attendance", y="predicted_cgpa", color="risk_level", title="Attendance vs. Predicted CGPA")
                st.plotly_chart(fig_scat, use_container_width=True)

            corr = dataset_df.corr(numeric_only=True)
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Feature Correlation Heatmap")
            st.plotly_chart(fig_corr, use_container_width=True)

        with tab_ml:
            st.subheader("Classifier & Regressor Performance Metrics")
            st.json(model_metrics)

    elif menu == "📜 All Student Records":
        render_header("📜 Saved Course Evaluation Records", "All evaluations logged across students and courses.")
        conn = sqlite3.connect("users.db")
        all_df = pd.read_sql_query("SELECT * FROM prediction_history ORDER BY id DESC", conn)
        conn.close()
        st.dataframe(all_df, use_container_width=True)

    elif menu == "🎓 5-Pillar Thesis Overview":
        render_header("🎓 Academic Thesis Architecture (5 Pillars)", "End-to-end system methodology for defense presentation.")
        st.markdown("""
        - **Pillar 1 — Problem Solving:** Proactive dropout/failure prevention via dual regression (CGPA) and multi-class risk stratification.
        - **Pillar 2 — Dataset:** 800-student university dataset covering attendance, prior CGPA, study/social habits, CT (/15), Midterm (/25), and instructor ratings.
        - **Pillar 3 — EDA & Preprocessing:** Correlation analysis, `StandardScaler` feature scaling, and domain-driven `study_balance` feature engineering.
        - **Pillar 4 — Modeling:** Multi-algorithm benchmarking (**Random Forest**, **XGBoost**, **SVM**) with serialized `.pkl` artifacts.
        - **Pillar 5 — XAI Contribution:** Perturbation-based local feature attribution + interactive 'What-If' counterfactual simulator.
        """)