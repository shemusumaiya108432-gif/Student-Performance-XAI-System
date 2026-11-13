import streamlit as st

def apply_custom_css():
    css = (
        "<style>"
        ".hero-banner { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 55%, #4338ca 100%); color: white; padding: 28px 32px; border-radius: 16px; text-align: center; margin-bottom: 24px; box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.35); }"
        ".hero-banner h1 { color: white !important; font-size: 30px; margin-bottom: 6px; font-weight: 800; }"
        ".hero-banner p { color: #e0e7ff; font-size: 15px; margin: 0; }"
        ".top-navbar { background: linear-gradient(90deg, #f8fafc 0%, #eef2ff 100%); border: 1px solid #e2e8f0; padding: 12px 20px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center; margin-bottom: 18px; }"
        ".role-pill-student { background-color: #dbeafe; color: #1e40af; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; }"
        ".role-pill-teacher { background-color: #fef3c7; color: #92400e; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: 700; }"
        ".notif-unread { background-color: #eef2ff; border-left: 4px solid #4338ca; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; font-size: 14px; }"
        ".notif-read { background-color: #f8fafc; border-left: 4px solid #cbd5e1; padding: 10px 14px; border-radius: 8px; margin-bottom: 8px; font-size: 14px; color: #64748b; }"
        ".course-header-box { background-color: #ffffff; padding: 16px 20px; border-radius: 12px; border: 1px solid #e5e7eb; display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }"
        ".course-code-badge { background-color: #312e81; color: white; padding: 5px 14px; border-radius: 8px; font-weight: 700; font-size: 14px; margin-right: 12px; }"
        ".course-title-text { font-size: 18px; font-weight: 600; color: #1f2937; }"
        ".credit-badge { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 5px 12px; border-radius: 20px; font-size: 13px; font-weight: 600; color: #334155; }"
        ".assess-card { background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }"
        ".assess-card-highlight { background-color: #eef2ff; border: 1.5px solid #c7d2fe; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px; }"
        ".assess-top { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }"
        ".assess-tag { background-color: #e2e8f0; color: #334155; padding: 3px 9px; border-radius: 6px; font-weight: 700; font-size: 12px; margin-right: 10px; }"
        ".assess-name { font-size: 15px; font-weight: 500; color: #1e293b; }"
        ".assess-score { font-size: 16px; font-weight: 700; color: #0f172a; }"
        ".assess-max { font-size: 13px; font-weight: 400; color: #64748b; }"
        ".progress-row { display: flex; align-items: center; gap: 12px; }"
        ".progress-bg { flex-grow: 1; background-color: #e2e8f0; height: 8px; border-radius: 999px; overflow: hidden; }"
        ".progress-fill { background: linear-gradient(90deg, #312e81 0%, #4f46e5 100%); height: 100%; border-radius: 999px; }"
        ".progress-pct { font-size: 13px; font-weight: 600; color: #475569; min-width: 38px; text-align: right; }"
        "</style>"
    )
    st.markdown(css, unsafe_allow_html=True)

def render_assessment_card(tag, title, score, max_score, highlight=False):
    pct = int(round((score / max_score) * 100)) if max_score > 0 else 0
    card_class = "assess-card-highlight" if highlight else "assess-card"
    score_disp = f"{int(score)}" if score == int(score) else f"{score:.1f}"
    html = (
        f'<div class="{card_class}"><div class="assess-top">'
        f'<div><span class="assess-tag">{tag}</span><span class="assess-name">{title}</span></div>'
        f'<div><span class="assess-score">{score_disp}</span> <span class="assess-max">/ {max_score}</span></div>'
        f'</div><div class="progress-row"><div class="progress-bg">'
        f'<div class="progress-fill" style="width: {min(100, pct)}%;"></div></div>'
        f'<div class="progress-pct">{pct}%</div></div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)