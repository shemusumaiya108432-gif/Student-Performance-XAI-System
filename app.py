import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import plotly.express as px
import plotly.graph_objects as go
from ui_components import apply_custom_css, render_assessment_card
from auth import (
    init_db, register_user, authenticate_user,
    get_all_students, get_courses_by_student_id, get_all_course_marks,
    upsert_course_marks, get_notifications, mark_notifications_read,
    add_notification, get_student_auto_summary
)

st.set_page_config(page_title="Academic AI Portal", page_icon="🎓", layout="wide")
init_db()
apply_custom_css()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_info = None

if "flash_msg" not in st.session_state:
    st.session_state.flash_msg = None

if st.session_state.flash_msg:
    msg_text, msg_icon = st.session_state.flash_msg
    st.toast(msg_text, icon=msg_icon)
    st.session_state.flash_msg = None

@st.cache_resource
def load_all():
    df = pd.read_csv("student_data.csv")
    reg_model = joblib.load("cgpa_model.pkl")
    risk_model = joblib.load("risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("model_metrics.json", "r") as f:
        metrics = json.load(f)
    return df, reg_model, risk_model, scaler, metrics

df, reg_model, risk_model, scaler, metrics = load_all()

def predict_student(att, prev_cgpa, study_hrs, social_hrs, tuition_val, quiz, mid, t_rating):
    study_bal = study_hrs - (social_hrs * 0.4)
    inp = np.array([[att, prev_cgpa, study_hrs, social_hrs, tuition_val, quiz, mid, t_rating, study_bal]])
    inp_scaled = scaler.transform(inp)
    pred_cgpa = round(float(reg_model.predict(inp_scaled)[0]), 2)
    risk_cls = int(risk_model.predict(inp_scaled)[0])
    risk_probs = risk_model.predict_proba(inp_scaled)[0]
    classes = list(risk_model.classes_)
    safe_prob = risk_probs[classes.index(0)] * 100 if 0 in classes else 0.0
    high_risk_prob = risk_probs[classes.index(2)] * 100 if 2 in classes else max(0.0, (3.0 - pred_cgpa) * 50)
    return pred_cgpa, risk_cls, safe_prob, min(99.0, high_risk_prob)

def generate_student_course_review(code, ct_a, pres, assign, mid, att):
    tips = []
    if att < 75:
        tips.append(f"📅 **Attendance Boost:** Current attendance is **{att:.0f}%**. Attend all remaining classes to cross **80%+** and avoid exam bar penalties.")
    if ct_a < 10.5:
        tips.append(f"📝 **Class Test Improvement:** Your CT average is **{ct_a:.1f}/15**. Review lecture slides weekly and practice short analytical questions.")
    if pres < 5.5:
        tips.append(f"🎤 **Presentation Polish:** Scored **{pres:.1f}/8**. Structure slides clearly and rehearse Q&A before viva/presentation.")
    if assign < 3.5:
        tips.append(f"📂 **Assignment Accuracy:** Scored **{assign:.1f}/5**. Submit clean, well-documented code/reports before deadlines.")
    if mid < 17.5:
        tips.append(f"📖 **Midterm Recovery:** Midterm score is **{mid:.1f}/25**. Solve previous years' final questions and consult your course teacher during office hours.")
    if not tips:
        tips.append(f"🌟 **Excellent Consistency in {code}!** Maintain your current study routine, focus on past final papers, and aim for an **A+**.")
    return tips

def generate_teacher_advice(pcgpa, rcls, att, study_hrs, social_hrs, ct_avg, mid, tr_in):
    advices = []
    if pcgpa < 2.50 or rcls == 2:
        advices.append("🚨 **Immediate Academic Counseling:** Schedule a mandatory 1-on-1 counseling session this week and notify the student's academic advisor.")
    elif pcgpa < 3.00 or rcls == 1:
        advices.append("⚠️ **Bi-Weekly Progress Check:** Monitor attendance and upcoming quiz submissions closely over the next 3 weeks.")
    else:
        advices.append("✅ **Positive Reinforcement:** Acknowledge consistent effort; encourage participation in research projects or peer-mentoring.")

    if att < 75:
        advices.append(f"📉 **Attendance Intervention ({att:.0f}%):** Issue an early attendance warning notice and verify if the student faces personal/health barriers.")
    if ct_avg < 9.5 or mid < 15.0:
        advices.append(f"📚 **Remedial Support Needed:** Recommend remedial tutorial classes or pair the student with a high-performing study group (CT: {ct_avg:.1f}/15, Mid: {mid:.1f}/25).")
    if social_hrs > study_hrs:
        advices.append(f"⏳ **Time-Management Coaching:** Social media usage ({social_hrs:.1f} hrs/day) exceeds daily study time ({study_hrs:.1f} hrs/day). Advise a structured daily study timetable.")
    if tr_in <= 2:
        advices.append("🤝 **Close Classroom Engagement:** Low teacher observation rating — actively involve the student in class Q&A and lab tasks.")
    return advices

# ==============================================================================
# 1. AUTHENTICATION PORTAL (LOGIN & REGISTRATION)
# ==============================================================================
if not st.session_state.logged_in:
    st.markdown(
        '<div class="hero-banner"><h1>🎓 Academic AI & Early Risk Identification Portal</h1>'
        '<p>Smart University Course Assessment Tracker • Explainable AI (XAI) • Real-Time Early Warning System</p></div>',
        unsafe_allow_html=True
    )
    col_info, col_auth = st.columns([1.15, 1.25], gap="large")
    with col_info:
        st.subheader("✨ Portal Highlights")
        st.info("👨‍🏫 **For Teachers:** Publish course-wise marks (`CT1–CT3`, `Presentation`, `Assignment`, `Mid Term`), auto-load student profiles, and send counseling advice.")
        st.success("🎓 **For Students:** View published course cards, track continuous assessment progress bars, get AI improvement reviews, and simulate final targets.")
        st.markdown("#### ⚡ Quick One-Click Demo Login")
        d1, d2 = st.columns(2)
        with d1:
            if st.button("🎓 Login as Demo Student", width="stretch"):
                u = authenticate_user("student1", "1234")
                if u:
                    st.session_state.logged_in = True
                    st.session_state.user_info = {"full_name": u[0], "username": u[1], "role": u[2], "student_id": u[3]}
                    st.session_state.flash_msg = (f"Welcome back, {u[0]}! Signed in as Student.", "🎓")
                    st.rerun()
        with d2:
            if st.button("👨‍🏫 Login as Demo Teacher", width="stretch"):
                u = authenticate_user("teacher1", "1234")
                if u:
                    st.session_state.logged_in = True
                    st.session_state.user_info = {"full_name": u[0], "username": u[1], "role": u[2], "student_id": u[3]}
                    st.session_state.flash_msg = (f"Welcome back, {u[0]}! Signed in as Teacher.", "👨‍🏫")
                    st.rerun()

    with col_auth:
        with st.container(border=True):
            t1, t2 = st.tabs(["🔐 Sign In", "📝 Create Account"])
            with t1:
                with st.form("login_form"):
                    lu = st.text_input("Username")
                    lp = st.text_input("Password", type="password")
                    if st.form_submit_button("Sign In ➔", type="primary", width="stretch"):
                        u = authenticate_user(lu, lp)
                        if u:
                            st.session_state.logged_in = True
                            st.session_state.user_info = {"full_name": u[0], "username": u[1], "role": u[2], "student_id": u[3]}
                            st.session_state.flash_msg = (f"Login Successful! Welcome, {u[0]}.", "✅")
                            st.rerun()
                        else:
                            st.toast("Invalid username or password!", icon="❌")
                            st.error("Invalid username or password.")

            with t2:
                r_role = st.selectbox("Register As:", ["Student", "Teacher"])
                with st.form("reg_form"):
                    rn = st.text_input("Full Name")
                    ru = st.text_input("Username")
                    rsid = st.text_input("Student ID (Required for Students)", placeholder="e.g., 232-35-696") if r_role == "Student" else ""
                    rp = st.text_input("Password", type="password")
                    rpc = st.text_input("Confirm Password", type="password")
                    submitted_reg = st.form_submit_button("Create Account ✨", type="primary", width="stretch")

                if submitted_reg:
                    if not rn.strip() or not ru.strip() or not rp:
                        st.error("⚠️ Please fill in all required fields.")
                    elif r_role == "Student" and not rsid.strip():
                        st.error("⚠️ Student ID is required for Student registration.")
                    elif rp != rpc:
                        st.error("❌ Passwords do not match.")
                    else:
                        ok, msg = register_user(rn, ru, rp, r_role, rsid)
                        if ok:
                            st.session_state.logged_in = True
                            st.session_state.user_info = {
                                "full_name": rn.strip(),
                                "username": ru.strip().lower(),
                                "role": r_role,
                                "student_id": rsid.strip().upper() if r_role == "Student" else ""
                            }
                            st.session_state.flash_msg = (f"🎉 Account created! Welcome, {rn.strip()}.", "🎉")
                            st.rerun()
                        else:
                            st.error(f"❌ {msg}")

# ==============================================================================
# 2. LOGGED-IN DASHBOARD
# ==============================================================================
else:
    user = st.session_state.user_info
    role = user["role"]

    top1, top2 = st.columns([4, 1])
    with top1:
        p_cls = "role-pill-student" if role == "Student" else "role-pill-teacher"
        id_txt = f" • ID: <b>{user['student_id']}</b>" if role == "Student" else ""
        st.markdown(
            f'<div class="top-navbar"><div><span style="font-size:18px;font-weight:700;color:#1e1b4b;">🎓 Academic AI Portal</span>'
            f'<span style="margin-left:12px;color:#475569;">Welcome, <b>{user["full_name"]}</b>{id_txt}</span></div>'
            f'<span class="{p_cls}">{role.upper()} ACCOUNT</span></div>',
            unsafe_allow_html=True
        )
    with top2:
        if st.button("🚪 Sign Out", width="stretch"):
            bname = user["full_name"]
            st.session_state.logged_in = False
            st.session_state.user_info = None
            st.session_state.pop("notified_once", None)
            st.session_state.flash_msg = (f"Signed out safely. Goodbye, {bname}!", "👋")
            st.rerun()

    st.sidebar.markdown(f"### 👤 {user['full_name']}")
    st.sidebar.caption(f"Role: **{role}**" + (f" | ID: **{user['student_id']}**" if role == "Student" else ""))
    st.sidebar.divider()

    # ==========================================================================
    # A. STUDENT PORTAL
    # ==========================================================================
    if role == "Student":
        notifs = get_notifications(user["student_id"])
        unread = sum(1 for n in notifs if n[3] == 0)
        if unread > 0 and "notified_once" not in st.session_state:
            st.toast(f"🔔 You have {unread} new notification(s)!", icon="🔔")
            st.session_state.notified_once = True

        with st.expander(f"🔔 Notification Center ({unread} Unread)", expanded=(unread > 0)):
            if not notifs:
                st.caption("No notifications yet.")
            else:
                if unread > 0 and st.button("✔ Mark All as Read"):
                    mark_notifications_read(user["student_id"])
                    st.toast("Notifications marked as read.", icon="✅")
                    st.rerun()
                for _, n_msg, n_time, n_read in notifs:
                    c_cls = "notif-unread" if n_read == 0 else "notif-read"
                    bdg = "🆕 " if n_read == 0 else ""
                    st.markdown(f"<div class='{c_cls}'>{bdg}<b>{n_msg}</b><br><span style='font-size:11px;color:#64748b;'>🕒 {n_time}</span></div>", unsafe_allow_html=True)

        st.sidebar.title("🎓 Student Menu")
        s_page = st.sidebar.radio("Navigate:", ["📚 My Course Marks & AI Review", "🎛️ 'What-If' Habit Simulator", "🎯 Safe Exam Benchmarks"])

        if s_page == "📚 My Course Marks & AI Review":
            st.subheader("📚 Continuous Assessment & AI Improvement Review")
            courses = get_courses_by_student_id(user["student_id"])
            if not courses:
                st.info(f"No course marks published yet for **{user['student_id']}**. Once your teacher publishes marks, your course cards and AI review will appear here automatically!")
            else:
                ov = []
                for c in courses:
                    ct_v = [x for x in [c[5], c[6], c[7]] if x > 0]
                    ct_a = round(sum(ct_v) / len(ct_v), 2) if ct_v else 0.0
                    tot = round(ct_a + c[8] + c[9] + c[10], 2)
                    stt = "🟢 Safe" if tot >= 35 and c[11] >= 75 else ("🟡 Moderate" if tot >= 26 else "🔴 At-Risk")
                    ov.append({"Code": c[2], "Title": c[3], "Credits": f"{c[4]:.0f} Cr", "CT Avg (/15)": ct_a, "Pres (/8)": c[8], "Assign (/5)": c[9], "Mid (/25)": c[10], "Total (/53)": tot, "Attendance": f"{c[11]:.0f}%", "Status": stt, "By": c[12]})
                st.dataframe(pd.DataFrame(ov), width="stretch", hide_index=True)
                st.divider()

                v_mode = st.radio("Display Mode:", ["Show All Courses Together", "Select Single Course"], horizontal=True)
                if v_mode == "Show All Courses Together":
                    sel_c = courses
                else:
                    chosen_c = st.selectbox("Choose Course:", [f"{x[2]} - {x[3]}" for x in courses])
                    sel_c = [c for c in courses if f"{c[2]} - {c[3]}" == chosen_c]

                for c in sel_c:
                    _, _, code, cname, cr, ct1, ct2, ct3, pres, assign, mid, att, pub_by = c
                    ct_v = [x for x in [ct1, ct2, ct3] if x > 0]
                    ct_a = round(sum(ct_v) / len(ct_v), 2) if ct_v else 0.0

                    col_card, col_ai = st.columns([1.3, 1.05])
                    with col_card:
                        st.markdown(f'<div class="course-header-box"><div><span class="course-code-badge">{code}</span><span class="course-title-text">{cname}</span></div><div class="credit-badge">📚 {cr:.0f} Cr</div></div>', unsafe_allow_html=True)
                        render_assessment_card("CT1", "Class Test 1", ct1, 15)
                        render_assessment_card("CT2", "Class Test 2", ct2, 15)
                        render_assessment_card("CT3", "Class Test 3", ct3, 15)
                        render_assessment_card("CT", "Class Test Average", ct_a, 15, highlight=True)
                        render_assessment_card("PR", "Presentation", pres, 8)
                        render_assessment_card("AS", "Assignment", assign, 5)
                        render_assessment_card("MID", "Mid Term", mid, 25)
                        st.caption(f"✔ Published by: **{pub_by}**")

                    with col_ai:
                        st.subheader(f"🤖 AI Review & Strategy: {code}")
                        pre = ct_a + pres + assign + mid
                        if pre < 26 or att < 65:
                            st.error(f"🔴 **HIGH RISK in {code}:** Marks ({pre:.1f}/53) or attendance ({att:.0f}%) is low.")
                        elif pre < 35:
                            st.warning(f"🟡 **MODERATE RISK in {code}:** Pre-final total is **{pre:.1f}/53**.")
                        else:
                            st.success(f"🟢 **SAFE ZONE in {code}:** Strong performance (**{pre:.1f}/53**).")

                        m1, m2 = st.columns(2)
                        m1.metric("Pre-Final Total", f"{pre:.1f} / 53")
                        m2.metric("Attendance", f"{att:.0f}%")

                        with st.container(border=True):
                            st.markdown("#### 💡 How to Improve Your Result:")
                            review_tips = generate_student_course_review(code, ct_a, pres, assign, mid, att)
                            for tip in review_tips:
                                st.markdown(f"- {tip}")

                        st.markdown("**🎯 Final Exam Targets (Out of 47):**")
                        st.write(f"- **A+ (80%)**: Need **{max(0.0, 80.0 - pre):.1f} / 47** | **B+ (65%)**: Need **{max(0.0, 65.0 - pre):.1f} / 47** | **Pass (40%)**: Need **{max(0.0, 40.0 - pre):.1f} / 47**")

                        xdf = pd.DataFrame({"Component": ["CT Avg", "Presentation", "Assignment", "Mid Term"], "Pct": [(ct_a/15)*100, (pres/8)*100, (assign/5)*100, (mid/25)*100]})
                        fig_c = px.bar(xdf, x="Pct", y="Component", orientation="h", title=f"{code} Mastery (%)", range_x=[0, 100], color="Pct", color_continuous_scale="Blues")
                        fig_c.update_layout(height=230, margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_c, width="stretch")
                    st.divider()

        elif s_page == "🎛️ 'What-If' Habit Simulator":
            st.title("🎛️ 'What-If' Interactive Habit & CGPA Simulator")
            cl, cr = st.columns([1, 1.3])
            with cl:
                sp = st.slider("Previous CGPA", 2.0, 4.0, 2.75, 0.05)
                sa = st.slider("Target Attendance (%)", 40.0, 100.0, 85.0, 1.0)
                ss = st.slider("Daily Study Hours", 0.5, 10.0, 4.0, 0.5)
                so = st.slider("Daily Social Media (Hours)", 0.5, 8.0, 1.5, 0.5)
                sq = st.slider("Target CT Average (/15)", 0.0, 15.0, 11.5, 0.5)
                sm = st.slider("Target Mid Term (/25)", 0.0, 25.0, 19.0, 0.5)
                st_job = st.radio("Part-time Tuition?", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", horizontal=True)
            scgpa, _, _, _ = predict_student(sa, sp, ss, so, st_job, sq, sm, 3)
            with cr:
                st.metric("Simulated CGPA", f"{scgpa:.2f} / 4.00", f"{scgpa - sp:+.2f}")
                fig_g = go.Figure(go.Indicator(mode="gauge+number", value=scgpa, title={'text': "Projected CGPA"}, gauge={'axis': {'range': [0, 4.0]}, 'bar': {'color': "#312e81"}, 'steps': [{'range': [0, 2.5], 'color': "#fecaca"}, {'range': [2.5, 3.0], 'color': "#fef08a"}, {'range': [3.0, 4.0], 'color': "#bbf7d0"}]}))
                fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=10))
                st.plotly_chart(fig_g, width="stretch")

                with st.container(border=True):
                    st.markdown("#### 💡 AI Habit & Study Plan Review:")
                    if ss < 3.5:
                        st.markdown(f"- ⏱️ **Increase Study Hours:** Increasing daily self-study from **{ss:.1f} hrs** to **4.5+ hrs** can significantly lift your CGPA.")
                    if so > 2.5:
                        st.markdown(f"- 📵 **Cut Screen Distractions:** Reducing social media from **{so:.1f} hrs** to **under 1.5 hrs** frees up valuable revision time.")
                    if sa < 80:
                        st.markdown(f"- 🏫 **Attendance Target:** Aim for **85%+ attendance** (currently simulated at {sa:.0f}%).")
                    if sq < 11.0 or sm < 18.0:
                        st.markdown("- 📈 **Assessment Goal:** Target **12+/15** in CTs and **19+/25** in Midterm to comfortably cross **3.50+ CGPA**.")
                    if ss >= 3.5 and so <= 2.5 and sa >= 80 and sq >= 11.0 and sm >= 18.0:
                        st.markdown("- 🌟 **Optimal Habit Balance!** Your study routine and target scores are well-aligned for top academic performance.")

        elif s_page == "🎯 Safe Exam Benchmarks":
            st.title("🎯 Safe Zone Continuous Assessment Benchmarks")
            b1, b2, b3, b4, b5 = st.columns(5)
            b1.metric("CT Avg (/15)", "10.5+ / 15")
            b2.metric("Presentation (/8)", "6.0+ / 8")
            b3.metric("Assignment (/5)", "4.0+ / 5")
            b4.metric("Mid Term (/25)", "17.5+ / 25")
            b5.metric("Attendance", "80% +")

    # ==========================================================================
    # B. TEACHER PORTAL
    # ==========================================================================
    elif role == "Teacher":
        st.sidebar.title("👨‍🏫 Teacher Menu")
        t_page = st.sidebar.radio("Navigate:", ["1️⃣ Upload / Publish Course Marks", "2️⃣ Student Risk Evaluation & Advice", "3️⃣ Batch Class Prediction (CSV / Excel)", "4️⃣ Class Analytical Dashboard (EDA)", "5️⃣ Model Comparison (Thesis Defense)"])

        if t_page == "1️⃣ Upload / Publish Course Marks":
            st.subheader("📤 Publish Course Marks & Notify Students")
            tab_s, tab_b = st.tabs(["✍️ Single Student Entry", "📂 Bulk CSV/Excel Upload"])
            with tab_s:
                stus = get_all_students()
                choices = [f"{s[0]} ({s[1]})" for s in stus]
                c_s1, c_s2 = st.columns([1.2, 1])
                with c_s1:
                    sel_lbl = st.selectbox("Select Registered Student:", choices) if choices else ""
                    tid = sel_lbl.split(" (")[0] if sel_lbl else "STD-2026-01"
                with c_s2:
                    mid_in = st.text_input("Or Type Student ID Directly:", placeholder="e.g., 232-35-696")
                    if mid_in.strip():
                        tid = mid_in.strip().upper()

                with st.form("pub_form"):
                    r1, r2, r3 = st.columns([1, 2, 0.8])
                    cc = r1.text_input("Course Code", "DS 421")
                    cn = r2.text_input("Course Title", "Machine Learning Driven Data Analysis I (DS Major)")
                    ccr = r3.number_input("Credits", 1.0, 4.0, 2.0, 0.5)
                    st.markdown(f"**Marks for `{tid}`**")
                    k1, k2, k3 = st.columns(3)
                    ct1 = k1.number_input("CT1 (/15)", 0.0, 15.0, 12.0, 0.5)
                    pr = k1.number_input("Presentation (/8)", 0.0, 8.0, 6.5, 0.5)
                    ct2 = k2.number_input("CT2 (/15)", 0.0, 15.0, 11.5, 0.5)
                    asg = k2.number_input("Assignment (/5)", 0.0, 5.0, 4.5, 0.5)
                    ct3 = k3.number_input("CT3 (/15)", 0.0, 15.0, 13.0, 0.5)
                    md = k3.number_input("Mid Term (/25)", 0.0, 25.0, 19.0, 0.5)
                    att = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 1.0)
                    if st.form_submit_button("🚀 Publish Marks & Notify", type="primary", width="stretch"):
                        upsert_course_marks(tid, cc, cn, ccr, ct1, ct2, ct3, pr, asg, md, att, updated_by=user["full_name"])
                        st.toast(f"Marks published & notified {tid}!", icon="🔔")
                        st.success(f"✅ Published **{cc}** marks to **{tid}**!")

            with tab_b:
                tdf = pd.DataFrame([{"student_id": "STD-2026-01", "course_code": "DS 421", "course_name": "Machine Learning Driven Data Analysis I (DS Major)", "credits": 2, "ct1": 13, "ct2": 12, "ct3": 14, "presentation": 7, "assignment": 5, "midterm": 20, "attendance": 88}])
                st.download_button("📥 Download Bulk Template (CSV)", tdf.to_csv(index=False).encode("utf-8"), "course_marks_template.csv", "text/csv")
                bf = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], key="b_up")
                if bf:
                    bdf = pd.read_csv(bf) if bf.name.endswith(".csv") else pd.read_excel(bf)
                    st.dataframe(bdf, width="stretch")
                    if st.button("🚀 Publish All Rows", type="primary"):
                        for _, r in bdf.iterrows():
                            upsert_course_marks(str(r["student_id"]), str(r["course_code"]), str(r["course_name"]), float(r["credits"]), float(r["ct1"]), float(r["ct2"]), float(r["ct3"]), float(r["presentation"]), float(r["assignment"]), float(r["midterm"]), float(r["attendance"]), updated_by=user["full_name"])
                        st.toast("All rows published & notified!", icon="🔔")
                        st.success(f"✅ Published {len(bdf)} records!")

            st.divider()
            st.subheader("📋 Published Course Records")
            recs = get_all_course_marks()
            if recs:
                st.dataframe(pd.DataFrame(recs, columns=["Student ID", "Code", "Title", "Cr", "CT1", "CT2", "CT3", "Pres", "Assign", "Mid", "Att (%)", "By"]), width="stretch", hide_index=True)

        elif t_page == "2️⃣ Student Risk Evaluation & Advice":
            st.title("👨‍🏫 Student Evaluation, Teacher Rating & Counseling Advice")
            all_s = get_all_students()
            opts = ["-- Select Registered Student --"] + [f"{s[0]} - {s[1]}" for s in all_s]
            sc1, sc2 = st.columns([1.2, 1])
            chosen = sc1.selectbox("🔍 Choose Registered Student:", opts)
            t_query = sc2.text_input("Or Search by Name / Student ID:", placeholder="e.g., Sumaiya or 232-35-696")

            l_key = t_query.strip() if t_query.strip() else (chosen.split(" - ")[0] if chosen != "-- Select Registered Student --" else "")
            adata = get_student_auto_summary(l_key) if l_key else None

            if adata:
                disp_id = f"{adata['student_id']} ({adata['full_name']})"
                d_att, d_ct, d_mid = float(adata["attendance"]), float(adata["ct_avg"]), float(adata["midterm"])
                if adata["has_marks"]:
                    st.success(f"✅ Auto-loaded data for **{disp_id}** ({adata['course_count']} course(s))!")
                else:
                    st.info(f"ℹ️ Found **{disp_id}**, no marks uploaded yet.")
            else:
                disp_id = l_key if l_key else "STD-2026-01"
                d_att, d_ct, d_mid = 58.0, 7.5, 12.0

            with st.form("eval_form"):
                c1, c2, c3 = st.columns(3)
                sid_in = c1.text_input("Student Name / ID", value=disp_id)
                att_in = c1.number_input("Attendance (%)", 0.0, 100.0, d_att, 1.0)
                pc_in = c1.number_input("Previous CGPA", 0.0, 4.0, 2.75 if adata else 2.45, 0.05)
                sh_in = c2.number_input("Daily Study Hours", 0.0, 16.0, 2.5 if adata else 1.5, 0.5)
                so_in = c2.number_input("Social Media Hours", 0.0, 16.0, 3.0 if adata else 4.5, 0.5)
                tu_in = c2.selectbox("Part-time Tuition?", ["No", "Yes"] if adata else ["Yes", "No"])
                qa_in = c3.number_input("CT Average (/15)", 0.0, 15.0, d_ct, 0.5)
                md_in = c3.number_input("Mid Term (/25)", 0.0, 25.0, d_mid, 0.5)
                tr_in = c3.select_slider("⭐ Teacher Rating (1-5)", options=[1, 2, 3, 4, 5], value=3 if adata else 2)
                custom_note = st.text_input("✍️ Optional Teacher Advice / Remark to Send Student:", placeholder="e.g., Focus on Midterm revision and meet me during office hours.")
                send_n = st.checkbox("🔔 Send evaluation & advice alert to Student's Notification Inbox", value=True)
                ebtn = st.form_submit_button("🔮 Evaluate & Generate Counseling Advice", type="primary", width="stretch")

            if ebtn:
                pcgpa, rcls, sprob, hprob = predict_student(att_in, pc_in, sh_in, so_in, 1 if tu_in == "Yes" else 0, qa_in, md_in, tr_in)
                st.divider()
                if pcgpa < 2.50 or rcls == 2 or tr_in <= 2:
                    st.toast(f"🚨 High Risk Alert for {sid_in}!", icon="🚨")
                    st.error(f"🚨 **HIGH RISK ({sid_in}):** Failure/dropout risk detected ({hprob:.1f}%).")
                    st_str = "🔴 High Risk"
                elif pcgpa < 3.00 or rcls == 1:
                    st.warning(f"🟡 **MODERATE RISK ({sid_in}):** Monitoring needed.")
                    st_str = "🟡 Moderate Risk"
                else:
                    st.success(f"🟢 **SAFE ZONE ({sid_in}):** Performing well.")
                    st_str = "🟢 Safe"

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted CGPA", f"{pcgpa:.2f} / 4.00")
                m2.metric("Status", st_str)
                m3.metric("Teacher Rating", f"{'⭐' * tr_in} ({tr_in}/5)")
                m4.metric("Safe Probability", f"{sprob:.1f}%")

                st.subheader("🩺 AI Teacher Intervention & Counseling Advice")
                teacher_advices = generate_teacher_advice(pcgpa, rcls, att_in, sh_in, so_in, qa_in, md_in, tr_in)
                with st.container(border=True):
                    for adv in teacher_advices:
                        st.markdown(f"- {adv}")
                    if custom_note.strip():
                        st.info(f"📝 **Teacher's Personal Note:** {custom_note.strip()}")

                if send_n and adata:
                    top_advice = custom_note.strip() if custom_note.strip() else teacher_advices[0].replace("**", "")
                    add_notification(
                        adata["student_id"],
                        f"📋 Evaluation by {user['full_name']}: {st_str} | Predicted CGPA: {pcgpa:.2f} | Advice: {top_advice}"
                    )
                    st.success(f"📩 Evaluation & advice sent directly to **{adata['full_name']}**'s Notification Inbox!")

        elif t_page == "3️⃣ Batch Class Prediction (CSV / Excel)":
            st.title("📂 Batch Class Prediction & At-Risk Export")
            sdf = df[['attendance', 'prev_cgpa', 'study_hours', 'social_media_hours', 'tuition', 'quiz_avg', 'mid_score', 'teacher_rating']].head(10).copy()
            if sdf['quiz_avg'].max() > 15:
                sdf['quiz_avg'] = ((sdf['quiz_avg'] / 100.0) * 15.0).round(1)
            if sdf['mid_score'].max() > 25:
                sdf['mid_score'] = ((sdf['mid_score'] / 100.0) * 25.0).round(1)
            sdf.insert(0, 'student_id', [f"STD-{101+i}" for i in range(10)])
            st.download_button("📥 Download Sample CSV", sdf.to_csv(index=False).encode("utf-8"), "sample_class_data.csv", "text/csv")
            uf = st.file_uploader("Upload Class File", type=["csv", "xlsx"])
            if uf:
                bdf = pd.read_csv(uf) if uf.name.endswith(".csv") else pd.read_excel(uf)
                bdf['study_balance'] = bdf['study_hours'] - (bdf['social_media_hours'] * 0.4)
                fcols = ['attendance', 'prev_cgpa', 'study_hours', 'social_media_hours', 'tuition', 'quiz_avg', 'mid_score', 'teacher_rating', 'study_balance']
                Xb = scaler.transform(bdf[fcols])
                bdf['Predicted_CGPA'] = reg_model.predict(Xb).round(2)
                bdf['Risk_Status'] = [{0: "🟢 Safe", 1: "🟡 Moderate Risk", 2: "🔴 High Risk"}.get(r, "🟡 Moderate") for r in risk_model.predict(Xb)]
                st.dataframe(bdf, width="stretch")
                st.download_button("📥 Download Full Report", bdf.to_csv(index=False).encode("utf-8"), "full_report.csv", "text/csv")

        elif t_page == "4️⃣ Class Analytical Dashboard (EDA)":
            st.title("📊 Class & Dataset Analytical Dashboard")
            dv = df.copy()
            dv['Risk_Label'] = dv['risk_level'].map({0: "Safe", 1: "Moderate Risk", 2: "High Risk"})
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.pie(dv, names='Risk_Label', title="Risk Distribution", color='Risk_Label', color_discrete_map={"Safe": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#ef4444"}), width="stretch")
            c2.plotly_chart(px.scatter(dv, x="attendance", y="predicted_cgpa", color="Risk_Label", title="Attendance vs CGPA", color_discrete_map={"Safe": "#10b981", "Moderate Risk": "#f59e0b", "High Risk": "#ef4444"}), width="stretch")

        elif t_page == "5️⃣ Model Comparison (Thesis Defense)":
            st.title("🏆 Model Performance & Comparison")
            cdf = pd.DataFrame([{"Algorithm": k, "Accuracy (%)": v["Accuracy"], "F1-Score (%)": v["F1_Score"]} for k, v in metrics.items()])
            ct, cc = st.columns([1, 1.4])
            ct.table(cdf)
            fig = go.Figure(data=[
                go.Bar(name='Accuracy (%)', x=cdf['Algorithm'], y=cdf['Accuracy (%)'], marker_color='#3b82f6'),
                go.Bar(name='F1-Score (%)', x=cdf['Algorithm'], y=cdf['F1-Score (%)'], marker_color='#8b5cf6')
            ])
            fig.update_layout(barmode='group', yaxis=dict(range=[0, 105]))
            cc.plotly_chart(fig, width="stretch")