import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import json
import os
import requests
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
# Initialize Google AI only when needed, not during import
google_api_key = os.getenv("GOOGLE_API_KEY")
model_ai = None
if google_api_key:
    try:
        genai.configure(api_key=google_api_key)
        model_ai = genai.GenerativeModel("gemini-2.0-flash")
    except Exception as e:
        print(f"Warning: Could not initialize Google AI: {e}")

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="EduPulse | Student Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# DESIGN SYSTEM - Dark Professional Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-card: #111827;
    --bg-elevated: #1a2235;
    --accent-blue: #3b82f6;
    --accent-teal: #14b8a6;
    --accent-amber: #f59e0b;
    --danger: #ef4444;
    --warning: #f97316;
    --success: #22c55e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border: #1e293b;
}

* { font-family: 'DM Sans', sans-serif; }

.stApp {
    background: var(--bg-primary);
    color: var(--text-primary);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #080c16 !important;
    border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
section[data-testid="stSidebar"] .stRadio label { color: var(--text-secondary) !important; }

/* Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: all 0.2s ease;
}
.metric-card:hover { border-color: var(--accent-blue); transform: translateY(-2px); }
.metric-label { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }
.metric-value { font-size: 32px; font-weight: 600; color: var(--text-primary); font-family: 'DM Serif Display', serif; }
.metric-sub { font-size: 12px; color: var(--text-secondary); margin-top: 4px; }

/* Risk badges */
.badge-high { background: rgba(239,68,68,0.15); color: #ef4444; border: 1px solid rgba(239,68,68,0.3); padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 13px; }
.badge-medium { background: rgba(249,115,22,0.15); color: #f97316; border: 1px solid rgba(249,115,22,0.3); padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 13px; }
.badge-low { background: rgba(34,197,94,0.15); color: #22c55e; border: 1px solid rgba(34,197,94,0.3); padding: 4px 12px; border-radius: 20px; font-weight: 600; font-size: 13px; }

/* Hero header */
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 48px;
    background: linear-gradient(135deg, #3b82f6, #14b8a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.1;
}
.hero-sub { color: var(--text-secondary); font-size: 16px; margin-top: 8px; font-weight: 300; }

/* Student card */
.student-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

/* Chat bubble */
.chat-user { background: var(--accent-blue); color: white; border-radius: 18px 18px 4px 18px; padding: 12px 16px; margin: 8px 0 8px 40px; font-size: 14px; }
.chat-ai { background: var(--bg-elevated); border: 1px solid var(--border); border-radius: 18px 18px 18px 4px; padding: 12px 16px; margin: 8px 40px 8px 0; font-size: 14px; color: var(--text-primary); }

/* Section header */
.section-header { font-family: 'DM Serif Display', serif; font-size: 24px; color: var(--text-primary); margin-bottom: 4px; }
.section-sub { color: var(--text-secondary); font-size: 14px; margin-bottom: 20px; }

/* Alert banner */
.alert-banner { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.3); border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.success-banner { background: rgba(34,197,94,0.1); border: 1px solid rgba(34,197,94,0.3); border-radius: 8px; padding: 12px 16px; margin: 8px 0; }
.info-banner { background: rgba(59,130,246,0.1); border: 1px solid rgba(59,130,246,0.3); border-radius: 8px; padding: 12px 16px; margin: 8px 0; }

/* Streamlit overrides */
.stButton button {
    background: var(--accent-blue);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
}
.stButton button:hover { background: #2563eb; }
div[data-testid="stMetric"] { background: var(--bg-card); border-radius: 12px; padding: 16px; border: 1px solid var(--border); }
div[data-testid="stMetric"] label { color: var(--text-secondary) !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* Input fields */
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stSelectbox [data-baseweb="select"] { background: var(--bg-elevated) !important; }

/* Dataframe */
.stDataFrame { background: var(--bg-card); border-radius: 12px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { background: var(--bg-card); border-radius: 8px; padding: 4px; border: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { color: var(--text-secondary); }
.stTabs [aria-selected="true"] { color: var(--text-primary) !important; }

/* Plotly bg override */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

hr { border-color: var(--border); }

/* Divider */
.divider { height: 1px; background: var(--border); margin: 24px 0; }

/* Intervention log */
.intervention-item {
    background: var(--bg-elevated);
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-left: 3px solid var(--accent-teal);
}

/* Pulse animation for high risk */
@keyframes pulse-border {
    0% { box-shadow: 0 0 0 0 rgba(239,68,68,0.4); }
    70% { box-shadow: 0 0 0 8px rgba(239,68,68,0); }
    100% { box-shadow: 0 0 0 0 rgba(239,68,68,0); }
}
.high-risk-pulse { animation: pulse-border 2s infinite; }

/* Logo area */
.logo-container { display: flex; align-items: center; gap: 10px; padding: 8px 0 20px 0; }
.logo-icon { font-size: 28px; }
.logo-text { font-family: 'DM Serif Display', serif; font-size: 22px; color: var(--text-primary); }
.logo-tag { font-size: 10px; color: var(--accent-teal); letter-spacing: 2px; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INTERVENTION LOG (in-memory storage simulating a DB)
# ============================================================
if 'interventions' not in st.session_state:
    st.session_state.interventions = {}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'selected_student_chat' not in st.session_state:
    st.session_state.selected_student_chat = None

# ============================================================
# LOAD MODEL & DATA
# ============================================================
@st.cache_resource
def load_model():
    try:
        model_pack = joblib.load("../models/student_risk_model.pkl")
        return model_pack['model'], model_pack['feature_columns']
    except:
        try:
            model_pack = joblib.load("models/student_risk_model.pkl")
            return model_pack['model'], model_pack['feature_columns']
        except Exception as e:
            st.error(f"❌ Cannot load model: {e}")
            st.stop()

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("../outputs/students_labeled.csv")
    except:
        try:
            df = pd.read_csv("outputs/students_labeled.csv")
        except Exception as e:
            st.error(f"❌ Cannot load student data: {e}")
            st.stop()
    return df

@st.cache_resource
def load_model_pack():
    try:
        return joblib.load("../models/student_risk_model.pkl")
    except:
        return joblib.load("models/student_risk_model.pkl")

model, model_columns = load_model()
df = load_data().copy()
model_pack = load_model_pack()

# ============================================================
# COMPUTE RISK SCORES
# ============================================================
@st.cache_data
def compute_all_risk_scores(_df):
    drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out', 'dropout']
    drop_cols = [c for c in drop_cols if c in _df.columns]
    X_all = _df[[c for c in _df.columns if c not in drop_cols]]
    X_all = pd.get_dummies(X_all, drop_first=True)
    X_all = X_all.reindex(columns=model_columns, fill_value=0)
    scores = model.predict_proba(X_all)[:, 1]
    return scores

df['risk_score'] = compute_all_risk_scores(df)
df['risk_category'] = df['risk_score'].apply(
    lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
)

def predict_student_risk(student_row):
    drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out', 'dropout', 'risk_score', 'risk_category']
    data = student_row.drop(labels=[c for c in drop_cols if c in student_row.index], errors='ignore')
    X = pd.DataFrame([data])
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=model_columns, fill_value=0)
    return model.predict_proba(X)[0, 1]

def risk_badge(category):
    if category == 'High':
        return '<span class="badge-high">⚠ HIGH RISK</span>'
    elif category == 'Medium':
        return '<span class="badge-medium">◉ MEDIUM RISK</span>'
    else:
        return '<span class="badge-low">✓ LOW RISK</span>'

def get_student_summary(row):
    """Build a text summary of a student for the AI chatbot context."""
    return f"""
Student ID: {row.get('student_id', 'N/A')}
Department: {row.get('department', 'N/A')}
Age: {row.get('age', 'N/A')}
CGPA: {row.get('cgpa', 'N/A')}
Attendance Rate: {row.get('attendance_rate', 'N/A')}%
Backlogs: {row.get('Backlogs', 'N/A')}
Study Hours/Week: {row.get('study_hours_per_week', 'N/A')}
Assignments Submitted: {row.get('assignments_submitted', 'N/A')}
Projects Completed: {row.get('projects_completed', 'N/A')}
Family Income: ₹{row.get('family_income', 'N/A')}
Scholarship: {row.get('scholarship', 'N/A')}
Risk Score: {row.get('risk_score', 0):.2f} ({row.get('risk_category', 'N/A')} Risk)
Income Group: {row.get('income_group', 'N/A')}
Extra Curricular: {row.get('extra_curricular', 'N/A')}
"""

# ============================================================
# AI COUNSELOR FUNCTION
# ============================================================
def ask_ai_counselor(question, student_context, api_key, history):
    """Call Claude API for student counseling advice."""
    if not api_key:
        return "⚠️ Please enter your Anthropic API key in the sidebar to enable AI Counselor."

    messages = []
    # Add history (last 6 turns)
    for turn in history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": question})

    system_prompt = f"""You are EduPulse AI Counselor — an expert academic advisor and student success specialist embedded in a university dashboard.

You are analyzing the following student profile:
{student_context}

Your role:
- Provide specific, actionable counseling advice based on this student's data
- Identify root causes of academic risk (not just symptoms)
- Suggest concrete intervention strategies faculty can take TODAY
- Be empathetic but direct — this is a professional tool for educators
- Use the student's actual numbers (CGPA, attendance, backlogs) in your analysis
- Keep responses concise and structured (use bullet points when listing actions)
- Never be generic — every response must reference this specific student's situation

Format your responses professionally. Use "Student" to refer to them (don't make up names)."""

    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 1024,
                "system": system_prompt,
                "messages": messages
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        elif response.status_code == 401:
            return "❌ Invalid API key. Please check your Anthropic API key in the sidebar."
        else:
            return f"❌ API Error ({response.status_code}): {response.json().get('error', {}).get('message', 'Unknown error')}"
    except requests.exceptions.ConnectionError:
        return "❌ Network error. Check your internet connection or API access."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def generate_email_alert(student_row, api_key):
    """Generate a professional alert email for a student."""
    if not api_key:
        return generate_fallback_email(student_row)
    
    context = get_student_summary(student_row)
    try:
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 600,
                "messages": [{
                    "role": "user",
                    "content": f"""Write a professional academic alert email to the student based on this profile:
{context}

The email should:
- Be from the Academic Counseling Office
- Mention their specific concerns (attendance, CGPA, backlogs as applicable)
- Invite them for a counseling appointment
- Be warm but serious
- Be concise (max 200 words)
- Subject line first, then email body

Format:
Subject: [subject here]

[email body here]"""
                }]
            },
            timeout=20
        )
        if response.status_code == 200:
            return response.json()['content'][0]['text']
        else:
            return generate_fallback_email(student_row)
    except:
        return generate_fallback_email(student_row)

def generate_fallback_email(row):
    concerns = []
    if row.get('attendance_rate', 100) < 75:
        concerns.append(f"attendance rate of {row.get('attendance_rate', 0):.0f}%")
    if row.get('Backlogs', 0) > 0:
        concerns.append(f"{int(row.get('Backlogs', 0))} active backlog(s)")
    if row.get('cgpa', 10) < 5:
        concerns.append(f"CGPA of {row.get('cgpa', 0):.1f}")
    concern_text = ", ".join(concerns) if concerns else "declining academic performance"
    return f"""Subject: Academic Support — Counseling Appointment Request

Dear Student (ID: {row.get('student_id', 'N/A')}),

I hope this message finds you well. Our academic monitoring system has flagged some areas of concern in your academic profile, specifically your {concern_text}.

We care deeply about your success, and we would like to invite you for a counseling session to understand any challenges you may be facing and explore support options available to you.

Please visit the Academic Counseling Office at your earliest convenience or reply to this email to schedule an appointment.

Remember, seeking help early is a sign of strength, and we are here to support you every step of the way.

Warm regards,
Academic Counseling Office
EduPulse Student Success Team"""

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div class="logo-container">
        <div class="logo-icon">🎓</div>
        <div>
            <div class="logo-text">EduPulse</div>
            <div class="logo-tag">Student Intelligence Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["📊 Command Center", "🔍 Student Deep Dive", "🤖 AI Counselor", "📋 Intervention Tracker", "📈 Analytics Hub", "📤 Batch Analysis"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**⚙️ Configuration**")
    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Required for AI Counselor & Email Generator"
    )

    if api_key:
        st.markdown('<div class="success-banner" style="font-size:12px;">✅ AI features enabled</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="info-banner" style="font-size:12px;">💡 Add API key for AI features</div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Quick stats in sidebar
    high_risk_n = len(df[df['risk_category'] == 'High'])
    st.markdown(f"""
    <div style="font-size:12px; color: #94a3b8;">
        <div style="margin-bottom:8px;">📊 <strong style="color:#f1f5f9;">{len(df):,}</strong> Total Students</div>
        <div style="margin-bottom:8px;">🔴 <strong style="color:#ef4444;">{high_risk_n}</strong> High Risk</div>
        <div style="margin-bottom:8px;">🟡 <strong style="color:#f97316;">{len(df[df['risk_category']=='Medium'])}</strong> Medium Risk</div>
        <div>🟢 <strong style="color:#22c55e;">{len(df[df['risk_category']=='Low'])}</strong> Low Risk</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: COMMAND CENTER (DASHBOARD)
# ============================================================
if page == "📊 Command Center":
    
    st.markdown("""
    <div style="padding: 8px 0 24px 0;">
        <div class="hero-title">Student Success Command Center</div>
        <div class="hero-sub">Real-time academic risk intelligence across your institution</div>
    </div>
    """, unsafe_allow_html=True)

    # KPI Row
    total = len(df)
    high_risk = len(df[df['risk_category'] == 'High'])
    med_risk = len(df[df['risk_category'] == 'Medium'])
    low_risk = len(df[df['risk_category'] == 'Low'])
    avg_attendance = df['attendance_rate'].mean()
    avg_cgpa = df['cgpa'].mean()
    dropout_actual = df['dropout'].sum() if 'dropout' in df.columns else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("👥 Total Students", f"{total:,}", help="Total enrolled students in dataset")
    col2.metric("🔴 High Risk", f"{high_risk}", f"{high_risk/total*100:.1f}% of total", delta_color="inverse")
    col3.metric("📅 Avg Attendance", f"{avg_attendance:.1f}%", help="Across all students")
    col4.metric("📝 Avg CGPA", f"{avg_cgpa:.2f}", help="Average CGPA across cohort")
    col5.metric("⚡ Actual Dropouts", f"{dropout_actual}", help="Historical dropout count in data")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Charts Row
    col_l, col_c, col_r = st.columns([1, 1, 1])

    with col_l:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = df['risk_category'].value_counts()
        fig_donut = go.Figure(data=[go.Pie(
            labels=risk_counts.index.tolist(),
            values=risk_counts.values.tolist(),
            hole=0.65,
            marker_colors=['#ef4444', '#f97316', '#22c55e'],
            textinfo='label+percent',
            textfont_color='white'
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            height=220,
            annotations=[dict(text=f"{high_risk/total*100:.0f}%<br><span style='font-size:10px'>High Risk</span>", x=0.5, y=0.5, font_size=18, font_color='#ef4444', showarrow=False)]
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_c:
        st.markdown('<div class="section-header">Risk by Department</div>', unsafe_allow_html=True)
        dept_risk = df.groupby('department')['risk_score'].mean().sort_values(ascending=True).tail(8)
        fig_dept = px.bar(
            x=dept_risk.values, y=dept_risk.index,
            orientation='h',
            color=dept_risk.values,
            color_continuous_scale=['#22c55e', '#f97316', '#ef4444'],
            labels={'x': 'Avg Risk Score', 'y': ''}
        )
        fig_dept.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0), height=220,
            coloraxis_showscale=False,
            xaxis=dict(color='#94a3b8', gridcolor='#1e293b'),
            yaxis=dict(color='#94a3b8')
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-header">CGPA vs Risk</div>', unsafe_allow_html=True)
        fig_scatter = px.scatter(
            df.sample(min(300, len(df))), x='cgpa', y='risk_score',
            color='risk_category',
            color_discrete_map={'High': '#ef4444', 'Medium': '#f97316', 'Low': '#22c55e'},
            opacity=0.7, size_max=8
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0), height=220,
            showlegend=False,
            xaxis=dict(color='#94a3b8', gridcolor='#1e293b', title='CGPA'),
            yaxis=dict(color='#94a3b8', gridcolor='#1e293b', title='Risk Score')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Urgent Action Table
    st.markdown('<div class="section-header">🚨 Immediate Action Required</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top 15 students needing counselor intervention today</div>', unsafe_allow_html=True)

    top15 = df.nlargest(15, 'risk_score')[['student_id', 'department', 'cgpa', 'attendance_rate', 'Backlogs', 'risk_score', 'risk_category']].copy()
    top15['risk_score'] = top15['risk_score'].apply(lambda x: f"{x:.3f}")
    top15['attendance_rate'] = top15['attendance_rate'].apply(lambda x: f"{x:.0f}%")
    top15['cgpa'] = top15['cgpa'].apply(lambda x: f"{x:.1f}")
    top15.columns = ['Student ID', 'Department', 'CGPA', 'Attendance', 'Backlogs', 'Risk Score', 'Category']

    st.dataframe(top15, use_container_width=True, hide_index=True,
        column_config={
            "Risk Score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.3f"),
            "Category": st.column_config.TextColumn("Risk Level")
        }
    )

# ============================================================
# PAGE: STUDENT DEEP DIVE
# ============================================================
elif page == "🔍 Student Deep Dive":

    st.markdown('<div class="hero-title">Student Deep Dive</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Individual student analysis, profiling and intervention planning</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Search
    col_search, col_filter = st.columns([2, 1])
    with col_search:
        student_ids = df['student_id'].astype(str).tolist()
        selected_id = st.selectbox("🔍 Select Student", student_ids, help="Search by Student ID")

    with col_filter:
        dept_filter = st.selectbox("Filter by Department", ['All'] + sorted(df['department'].unique().tolist()))

    if dept_filter != 'All':
        display_df = df[df['department'] == dept_filter]
    else:
        display_df = df

    student_row = df[df['student_id'].astype(str) == selected_id].iloc[0]
    risk_score = student_row['risk_score']
    risk_cat = student_row['risk_category']
    risk_color = '#ef4444' if risk_cat == 'High' else '#f97316' if risk_cat == 'Medium' else '#22c55e'

    # Profile + Risk gauge
    col_profile, col_gauge, col_ai_quick = st.columns([1.2, 0.9, 0.9])

    with col_profile:
        st.markdown(f"""
        <div class="metric-card" style="text-align:left; padding: 24px;">
            <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Student Profile</div>
            <div style="font-family:'DM Serif Display', serif; font-size:20px; color:#f1f5f9; margin-bottom:16px;">ID: {student_row.get('student_id', 'N/A')}</div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Department</div>
                    <div style="font-size:14px; color:#f1f5f9; font-weight:500;">{student_row.get('department', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Age</div>
                    <div style="font-size:14px; color:#f1f5f9; font-weight:500;">{student_row.get('age', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">CGPA</div>
                    <div style="font-size:14px; color:#f1f5f9; font-weight:500;">{student_row.get('cgpa', 0):.2f}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Attendance</div>
                    <div style="font-size:14px; color:#{'ef4444' if student_row.get('attendance_rate', 100) < 75 else 'f1f5f9'}; font-weight:500;">{student_row.get('attendance_rate', 0):.0f}%</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Backlogs</div>
                    <div style="font-size:14px; color:#{'ef4444' if student_row.get('Backlogs', 0) > 0 else 'f1f5f9'}; font-weight:500;">{int(student_row.get('Backlogs', 0))}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Scholarship</div>
                    <div style="font-size:14px; color:#f1f5f9; font-weight:500;">{student_row.get('scholarship', 'N/A')}</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Study Hours/Wk</div>
                    <div style="font-size:14px; color:#f1f5f9; font-weight:500;">{student_row.get('study_hours_per_week', 0):.0f}h</div>
                </div>
                <div>
                    <div style="font-size:11px; color:#94a3b8;">Income Group</div>
                    <div style="font-size:14px; color:#f1f5f9; font-weight:500;">{str(student_row.get('income_group', 'N/A')).title()}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={'suffix': '', 'font': {'size': 36, 'color': risk_color}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score", 'font': {'color': '#94a3b8', 'size': 14}},
            gauge={
                'axis': {'range': [0, 1], 'tickcolor': '#94a3b8', 'tickfont': {'color': '#94a3b8', 'size': 10}},
                'bar': {'color': risk_color},
                'bgcolor': '#1a2235',
                'bordercolor': '#1e293b',
                'steps': [
                    {'range': [0, 0.4], 'color': 'rgba(34,197,94,0.1)'},
                    {'range': [0.4, 0.7], 'color': 'rgba(249,115,22,0.1)'},
                    {'range': [0.7, 1], 'color': 'rgba(239,68,68,0.1)'}
                ],
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(17,24,39,1)', plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=30, b=10), height=220,
            font={'color': '#94a3b8'}
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown(f"<div style='text-align:center;'>{risk_badge(risk_cat)}</div>", unsafe_allow_html=True)

    with col_ai_quick:
        st.markdown("""
        <div class="metric-card" style="text-align:left; height: 100%;">
            <div style="font-size:12px; color:#94a3b8; text-transform:uppercase; letter-spacing:1px; margin-bottom:12px;">Risk Signals</div>
        """, unsafe_allow_html=True)

        signals = []
        if student_row.get('attendance_rate', 100) < 65:
            signals.append(("🔴", "Critical attendance", f"{student_row.get('attendance_rate', 0):.0f}%"))
        elif student_row.get('attendance_rate', 100) < 75:
            signals.append(("🟡", "Low attendance", f"{student_row.get('attendance_rate', 0):.0f}%"))
        if student_row.get('Backlogs', 0) > 2:
            signals.append(("🔴", "Multiple backlogs", f"{int(student_row.get('Backlogs', 0))} subjects"))
        elif student_row.get('Backlogs', 0) > 0:
            signals.append(("🟡", "Has backlog(s)", f"{int(student_row.get('Backlogs', 0))} subject"))
        if student_row.get('cgpa', 10) < 4:
            signals.append(("🔴", "Very low CGPA", f"{student_row.get('cgpa', 0):.1f}"))
        elif student_row.get('cgpa', 10) < 6:
            signals.append(("🟡", "Below average CGPA", f"{student_row.get('cgpa', 0):.1f}"))
        if student_row.get('study_hours_per_week', 20) < 10:
            signals.append(("🟡", "Low study hours", f"{student_row.get('study_hours_per_week', 0):.0f}h/week"))
        if student_row.get('income_group', '') == 'low':
            signals.append(("🔵", "Financial risk", "Low income"))

        if not signals:
            signals.append(("🟢", "No critical signals", "Student appears stable"))

        for icon, label, value in signals[:5]:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; padding:6px 0; border-bottom:1px solid #1e293b;">
                <span style="font-size:13px;">{icon} {label}</span>
                <span style="font-size:12px; color:#94a3b8; font-weight:500;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Feature importance for this student
    col_feat, col_email = st.columns([1.4, 0.6])

    with col_feat:
        st.markdown('<div class="section-header">🧠 What\'s Driving the Risk</div>', unsafe_allow_html=True)
        if 'feature_importance' in model_pack:
            fi = model_pack['feature_importance'].head(10)
            fig_fi = px.bar(
                fi, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale=['#14b8a6', '#3b82f6', '#ef4444']
            )
            fig_fi.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                margin=dict(l=0, r=0, t=10, b=0), height=280,
                coloraxis_showscale=False,
                xaxis=dict(color='#94a3b8', gridcolor='#1e293b'),
                yaxis=dict(color='#94a3b8')
            )
            st.plotly_chart(fig_fi, use_container_width=True)

    with col_email:
        st.markdown('<div class="section-header">📧 Alert Email</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Auto-generate a personalized alert email</div>', unsafe_allow_html=True)
        if st.button("✉️ Generate Alert Email", use_container_width=True):
            with st.spinner("Generating email..."):
                email_text = generate_email_alert(student_row, api_key)
            st.session_state[f'email_{selected_id}'] = email_text

        if f'email_{selected_id}' in st.session_state:
            st.text_area("Generated Email", st.session_state[f'email_{selected_id}'], height=200)
            st.download_button("📥 Download Email", st.session_state[f'email_{selected_id}'], file_name=f"alert_{selected_id}.txt")

# ============================================================
# PAGE: AI COUNSELOR
# ============================================================
elif page == "🤖 AI Counselor":

    st.markdown('<div class="hero-title">AI Counselor</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Have a deep conversation about any student with our AI-powered counseling assistant</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_sel, col_clear = st.columns([3, 1])
    with col_sel:
        student_ids = df['student_id'].astype(str).tolist()
        chat_student_id = st.selectbox("Select Student to Discuss", student_ids, key="chat_student_select")
    with col_clear:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    chat_row = df[df['student_id'].astype(str) == chat_student_id].iloc[0]
    student_context = get_student_summary(chat_row)

    # Show student mini card
    risk_c = chat_row['risk_category']
    rc_color = '#ef4444' if risk_c == 'High' else '#f97316' if risk_c == 'Medium' else '#22c55e'
    st.markdown(f"""
    <div style="background:#111827; border:1px solid #1e293b; border-left: 3px solid {rc_color}; border-radius:8px; padding:12px 16px; margin-bottom:16px; display:flex; gap:24px; align-items:center;">
        <div><span style="color:#94a3b8; font-size:11px;">STUDENT</span><br><span style="font-size:16px; font-weight:600;">ID: {chat_row.get('student_id')}</span></div>
        <div><span style="color:#94a3b8; font-size:11px;">DEPT</span><br><span style="font-size:14px;">{chat_row.get('department', 'N/A')}</span></div>
        <div><span style="color:#94a3b8; font-size:11px;">CGPA</span><br><span style="font-size:14px;">{chat_row.get('cgpa', 0):.1f}</span></div>
        <div><span style="color:#94a3b8; font-size:11px;">ATTENDANCE</span><br><span style="font-size:14px;">{chat_row.get('attendance_rate', 0):.0f}%</span></div>
        <div><span style="color:#94a3b8; font-size:11px;">RISK</span><br>{risk_badge(risk_c)}</div>
    </div>
    """, unsafe_allow_html=True)

    if not api_key:
        st.markdown("""
        <div class="info-banner">
            🤖 Add your Anthropic API key in the sidebar to enable the AI Counselor. Without it, you'll get basic template responses.
        </div>
        """, unsafe_allow_html=True)

    # Quick prompt suggestions
    st.markdown("**💡 Quick Prompts:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    quick_prompts = [
        "What are the main risk factors for this student?",
        "What intervention should I take this week?",
        "Is financial aid a concern for this student?",
        "How does this student compare to typical dropouts?"
    ]
    for i, (col, prompt) in enumerate(zip([qcol1, qcol2, qcol3, qcol4], quick_prompts)):
        if col.button(prompt[:35] + "...", key=f"qp_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.spinner("AI Counselor thinking..."):
                reply = ask_ai_counselor(prompt, student_context, api_key, st.session_state.chat_history[:-1])
            st.session_state.chat_history.append({"role": "assistant", "content": reply})
            st.rerun()

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f'<div class="chat-user">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input
    user_input = st.chat_input("Ask anything about this student...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("AI Counselor thinking..."):
            reply = ask_ai_counselor(user_input, student_context, api_key, st.session_state.chat_history[:-1])
        st.session_state.chat_history.append({"role": "assistant", "content": reply})
        st.rerun()

# ============================================================
# PAGE: INTERVENTION TRACKER
# ============================================================
elif page == "📋 Intervention Tracker":

    st.markdown('<div class="hero-title">Intervention Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Log and track all counseling actions and interventions for at-risk students</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_form, col_log = st.columns([1, 1.2])

    with col_form:
        st.markdown('<div class="section-header">📝 Log New Intervention</div>', unsafe_allow_html=True)
        
        with st.container():
            # Focus on high risk students
            high_risk_students = df[df['risk_category'] == 'High']['student_id'].astype(str).tolist()
            all_students = df['student_id'].astype(str).tolist()
            
            int_student = st.selectbox("Student ID", all_students, help="Select student")
            int_type = st.selectbox("Intervention Type", [
                "📞 Phone Call to Student",
                "💬 One-on-One Counseling Session",
                "📧 Email Alert Sent",
                "👪 Parent/Guardian Contact",
                "📚 Academic Support Arranged",
                "💰 Financial Aid Referral",
                "🏥 Mental Health Referral",
                "👥 Peer Mentoring Assigned",
                "📋 Academic Warning Issued",
                "✅ Follow-up Check-in"
            ])
            int_counselor = st.text_input("Counselor Name", placeholder="Your name")
            int_outcome = st.selectbox("Outcome", [
                "✅ Student responded positively",
                "⚠️ Student needs more support",
                "❌ Unable to reach student",
                "📅 Follow-up scheduled",
                "🔄 Ongoing monitoring",
                "✔️ Issue resolved"
            ])
            int_notes = st.text_area("Notes", placeholder="Additional details about the intervention...", height=100)

            if st.button("✅ Log Intervention", type="primary", use_container_width=True):
                if int_counselor:
                    sid = int_student
                    if sid not in st.session_state.interventions:
                        st.session_state.interventions[sid] = []
                    st.session_state.interventions[sid].append({
                        "type": int_type,
                        "counselor": int_counselor,
                        "outcome": int_outcome,
                        "notes": int_notes,
                        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
                    })
                    st.success(f"✅ Intervention logged for Student {sid}")
                else:
                    st.error("Please enter counselor name.")

    with col_log:
        st.markdown('<div class="section-header">📜 Recent Interventions</div>', unsafe_allow_html=True)
        
        total_interventions = sum(len(v) for v in st.session_state.interventions.values())
        students_with_logs = len(st.session_state.interventions)

        col_a, col_b = st.columns(2)
        col_a.metric("Total Logged", total_interventions)
        col_b.metric("Students Tracked", students_with_logs)
        
        st.markdown("<br>", unsafe_allow_html=True)

        if st.session_state.interventions:
            # Flatten all interventions
            all_ints = []
            for sid, items in st.session_state.interventions.items():
                for item in items:
                    all_ints.append({"Student": sid, **item})
            
            # Sort by most recent
            all_ints.reverse()
            
            for item in all_ints[:20]:
                st.markdown(f"""
                <div class="intervention-item">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div>
                            <div style="font-weight:600; font-size:14px; color:#f1f5f9;">{item['type']}</div>
                            <div style="font-size:12px; color:#94a3b8; margin-top:4px;">Student {item['Student']} • By {item['counselor']}</div>
                            <div style="font-size:12px; color:#14b8a6; margin-top:4px;">{item['outcome']}</div>
                            {f'<div style="font-size:11px; color:#64748b; margin-top:4px;">{item["notes"][:80]}...</div>' if item.get('notes') else ''}
                        </div>
                        <div style="font-size:11px; color:#64748b; white-space:nowrap;">{item['timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Export
            if st.button("📥 Export to CSV", use_container_width=True):
                export_df = pd.DataFrame(all_ints)
                csv = export_df.to_csv(index=False)
                st.download_button("Download CSV", csv, "interventions.csv", "text/csv")
        else:
            st.markdown("""
            <div style="text-align:center; padding:40px; color:#94a3b8;">
                <div style="font-size:40px; margin-bottom:12px;">📋</div>
                <div style="font-size:14px;">No interventions logged yet.</div>
                <div style="font-size:12px; margin-top:4px;">Log your first intervention using the form on the left.</div>
            </div>
            """, unsafe_allow_html=True)

# ============================================================
# PAGE: ANALYTICS HUB
# ============================================================
elif page == "📈 Analytics Hub":

    st.markdown('<div class="hero-title">Analytics Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Deep cohort analysis, trends, and predictive insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏫 Cohort Analysis", "📊 Feature Insights", "🎯 Model Performance"])

    with tab1:
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Risk Score Distribution by Department**")
            dept_box = px.box(df, x='department', y='risk_score', color='department',
                title="", points='outliers')
            dept_box.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False, height=300,
                xaxis=dict(color='#94a3b8', gridcolor='#1e293b'),
                yaxis=dict(color='#94a3b8', gridcolor='#1e293b')
            )
            dept_box.update_traces(marker_color='#3b82f6')
            st.plotly_chart(dept_box, use_container_width=True)

        with col_r:
            st.markdown("**Risk Distribution by Income Group**")
            income_risk = df.groupby('income_group')['risk_score'].mean().reset_index()
            income_risk.columns = ['Income Group', 'Avg Risk Score']
            fig_income = px.bar(income_risk, x='Income Group', y='Avg Risk Score',
                color='Avg Risk Score', color_continuous_scale=['#22c55e', '#f97316', '#ef4444'])
            fig_income.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False, height=300,
                xaxis=dict(color='#94a3b8'), yaxis=dict(color='#94a3b8', gridcolor='#1e293b')
            )
            st.plotly_chart(fig_income, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown("**Attendance Rate vs Risk Score**")
            fig_att = px.scatter(df, x='attendance_rate', y='risk_score',
                color='risk_category',
                color_discrete_map={'High': '#ef4444', 'Medium': '#f97316', 'Low': '#22c55e'},
                trendline='lowess', opacity=0.6)
            fig_att.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, showlegend=False,
                xaxis=dict(color='#94a3b8', gridcolor='#1e293b', title='Attendance %'),
                yaxis=dict(color='#94a3b8', gridcolor='#1e293b', title='Risk Score')
            )
            st.plotly_chart(fig_att, use_container_width=True)

        with col_r2:
            st.markdown("**Backlog Count vs Risk Category**")
            backlog_dist = df.groupby(['Backlogs', 'risk_category']).size().reset_index(name='count')
            fig_bl = px.bar(backlog_dist, x='Backlogs', y='count', color='risk_category',
                color_discrete_map={'High': '#ef4444', 'Medium': '#f97316', 'Low': '#22c55e'})
            fig_bl.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=280, legend_title='',
                xaxis=dict(color='#94a3b8', title='Number of Backlogs'),
                yaxis=dict(color='#94a3b8', gridcolor='#1e293b', title='Student Count')
            )
            st.plotly_chart(fig_bl, use_container_width=True)

    with tab2:
        st.markdown("**Top Factors Predicting Student Risk**")
        if 'feature_importance' in model_pack:
            fi = model_pack['feature_importance'].head(15)
            fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                color='Importance', color_continuous_scale=['#14b8a6', '#3b82f6', '#ef4444'])
            fig_fi.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=450, coloraxis_showscale=False,
                xaxis=dict(color='#94a3b8', gridcolor='#1e293b'),
                yaxis=dict(color='#94a3b8')
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("**Feature Correlations**")
        numeric_cols = ['cgpa', 'attendance_rate', 'Backlogs', 'study_hours_per_week', 
                       'assignments_submitted', 'projects_completed', 'family_income', 'risk_score']
        numeric_cols = [c for c in numeric_cols if c in df.columns]
        corr = df[numeric_cols].corr()
        fig_heat = px.imshow(corr, color_continuous_scale='RdBu_r', aspect='auto',
            zmin=-1, zmax=1, text_auto='.2f')
        fig_heat.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', height=400,
            font_color='#94a3b8'
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        if 'metrics' in model_pack:
            metrics = model_pack['metrics']
            col1, col2, col3 = st.columns(3)
            col1.metric("🎯 Accuracy", f"{metrics.get('accuracy', 0):.1%}")
            col2.metric("📊 ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
            col3.metric("⚙️ Model Type", "Random Forest")

        st.markdown("""
        <div class="info-banner" style="margin-top:16px;">
            <strong>Model Information:</strong> This platform uses a Random Forest Classifier trained on historical student data. 
            A risk score > 0.7 indicates HIGH risk. The model was trained with SMOTE balancing to handle class imbalance.
            SHAP values are used to explain individual predictions.
        </div>
        """, unsafe_allow_html=True)

        # Department-wise performance table
        st.markdown("**📋 Risk Summary by Department**")
        dept_summary = df.groupby('department').agg(
            Students=('student_id', 'count'),
            Avg_Risk=('risk_score', 'mean'),
            High_Risk=('risk_category', lambda x: (x == 'High').sum()),
            Avg_CGPA=('cgpa', 'mean'),
            Avg_Attendance=('attendance_rate', 'mean')
        ).reset_index()
        dept_summary.columns = ['Department', 'Students', 'Avg Risk', 'High Risk', 'Avg CGPA', 'Avg Attendance']
        dept_summary = dept_summary.sort_values('Avg Risk', ascending=False)
        dept_summary['Avg Risk'] = dept_summary['Avg Risk'].apply(lambda x: f"{x:.3f}")
        dept_summary['Avg CGPA'] = dept_summary['Avg CGPA'].apply(lambda x: f"{x:.2f}")
        dept_summary['Avg Attendance'] = dept_summary['Avg Attendance'].apply(lambda x: f"{x:.1f}%")
        st.dataframe(dept_summary, use_container_width=True, hide_index=True)

# ============================================================
# PAGE: BATCH ANALYSIS
# ============================================================
elif page == "📤 Batch Analysis":

    st.markdown('<div class="hero-title">Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload new student cohort data for instant risk assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col_upload, col_info = st.columns([1.5, 1])

    with col_upload:
        uploaded = st.file_uploader("📁 Upload Student CSV", type=['csv'],
            help="CSV should have same columns as training data")

        if uploaded:
            try:
                batch_df = pd.read_csv(uploaded)
                st.success(f"✅ Loaded {len(batch_df)} students")
                st.dataframe(batch_df.head(5), use_container_width=True)

                if st.button("🚀 Run Risk Assessment", type="primary", use_container_width=True):
                    with st.spinner("Analyzing cohort..."):
                        drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out', 'dropout', 'risk_score', 'risk_category']
                        drop_cols_present = [c for c in drop_cols if c in batch_df.columns]
                        id_cols = ['student_id'] if 'student_id' in batch_df.columns else []

                        X_batch = batch_df[[c for c in batch_df.columns if c not in drop_cols]]
                        X_batch = pd.get_dummies(X_batch, drop_first=True)
                        X_batch = X_batch.reindex(columns=model_columns, fill_value=0)
                        scores = model.predict_proba(X_batch)[:, 1]

                        result_df = batch_df[id_cols].copy() if id_cols else pd.DataFrame()
                        result_df['risk_score'] = scores
                        result_df['risk_category'] = result_df['risk_score'].apply(
                            lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
                        )
                        result_df['priority'] = result_df['risk_score'].rank(ascending=False).astype(int)
                        result_df = result_df.sort_values('risk_score', ascending=False)

                        st.session_state.batch_results = result_df

            except Exception as e:
                st.error(f"❌ Error: {e}")

    with col_info:
        st.markdown("""
        <div class="metric-card" style="text-align:left;">
            <div style="font-size:13px; font-weight:600; margin-bottom:12px; color:#f1f5f9;">📋 Expected Columns</div>
            <div style="font-size:12px; color:#94a3b8; line-height:2;">
                student_id • gender • department<br>
                scholarship • parental_education<br>
                extra_curricular • age • cgpa<br>
                attendance_rate • family_income<br>
                Backlogs • study_hours_per_week<br>
                assignments_submitted<br>
                projects_completed • total_activities<br>
                sports_participation • income_group
            </div>
        </div>
        """, unsafe_allow_html=True)

    if 'batch_results' in st.session_state:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header">📊 Batch Results</div>', unsafe_allow_html=True)

        results = st.session_state.batch_results
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Analyzed", len(results))
        c2.metric("High Risk", (results['risk_category'] == 'High').sum())
        c3.metric("Medium Risk", (results['risk_category'] == 'Medium').sum())
        c4.metric("Low Risk", (results['risk_category'] == 'Low').sum())

        st.dataframe(results, use_container_width=True, hide_index=True,
            column_config={
                "risk_score": st.column_config.ProgressColumn("Risk Score", min_value=0, max_value=1, format="%.3f"),
            }
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            csv = results.to_csv(index=False)
            st.download_button("📥 Download All Results", csv, "batch_risk_results.csv", "text/csv", use_container_width=True)
        with col_dl2:
            high_only = results[results['risk_category'] == 'High'].to_csv(index=False)
            st.download_button("🔴 Download High Risk Only", high_only, "high_risk_students.csv", "text/csv", use_container_width=True)

# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div style='text-align: center; color: #334155; padding: 32px; font-size: 12px; letter-spacing: 0.5px;'>
    <span style="color:#3b82f6; font-family: 'DM Serif Display', serif; font-size:14px;">EduPulse</span> 
    &nbsp;•&nbsp; Student Intelligence Platform &nbsp;•&nbsp; Powered by AI & Machine Learning
</div>
""", unsafe_allow_html=True)