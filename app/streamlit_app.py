import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Student Risk Dashboard",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR STYLING
# ============================================
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Risk level colors */
    .risk-high {
        color: #d62728;
        font-weight: bold;
        font-size: 28px;
    }
    .risk-medium {
        color: #ff7f0e;
        font-weight: bold;
        font-size: 28px;
    }
    .risk-low {
        color: #2ca02c;
        font-weight: bold;
        font-size: 28px;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# LOAD DATA (with caching for speed)
# ============================================
@st.cache_resource
def load_model():
    """Load the trained model"""
    try:
        model_pack = joblib.load("models/student_risk_model.pkl")
        return model_pack['model'], model_pack['feature_columns']
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()

@st.cache_data
def load_data():
    """Load student data"""
    try:
        return pd.read_csv("outputs/students_labeled.csv")
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        st.stop()

@st.cache_resource
def load_shap():
    """Load SHAP explanations"""
    # Try multiple fallbacks: shap_values (pickle) or feature_importance (joblib)
    try:
        import joblib as _joblib
        # Prefer shap values file if present
        try:
            with open("outputs/shap_values.pkl", "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

        # Fallback: feature importance saved by compute_shap (joblib)
        try:
            return _joblib.load("outputs/feature_importance.pkl")
        except Exception:
            return None
    except Exception:
        return None

# Load everything
model, model_columns = load_model()
df = load_data()
shap_data = load_shap()

# Helper to resolve column name variants present in the dataset
def get_col(series_or_df, *names):
    """Return the first name that exists in series_or_df (Series.index or DataFrame.columns)."""
    keys = None
    if hasattr(series_or_df, 'columns'):
        keys = set(series_or_df.columns)
    else:
        keys = set(series_or_df.index)

    for n in names:
        if n in keys:
            return n
    return None

# Resolve commonly used column names once so they are available globally
att_col = get_col(df, 'attendance_percent', 'attendance_rate')
back_col = get_col(df, 'num_backlogs', 'Backlogs')
marks_col = get_col(df, 'avg_internal_marks', 'cgpa')

# ============================================
# HELPER FUNCTIONS
# ============================================
def predict_risk(student_data):
    """Predict risk score for a student"""
    # Prepare data
    X = pd.DataFrame([student_data])
    X = pd.get_dummies(X, drop_first=True)
    X = X.reindex(columns=model_columns, fill_value=0)
    
    # Predict
    prob = model.predict_proba(X)[0, 1]
    return prob

def get_risk_category(score):
    """Convert score to category"""
    if score >= 0.7:
        return "HIGH RISK", "risk-high", "#d62728"
    elif score >= 0.4:
        return "MEDIUM RISK", "risk-medium", "#ff7f0e"
    else:
        return "LOW RISK", "risk-low", "#2ca02c"

def get_recommendations(student_row, shap_explanation):
    """Generate action recommendations based on risk factors"""
    recommendations = []
    
    # Check attendance
    if 'attendance_percent' in student_row.index:
        if student_row['attendance_percent'] < 65:
            recommendations.append("📞 **Urgent:** Schedule attendance counseling session")
        elif student_row['attendance_percent'] < 75:
            recommendations.append("📅 **Monitor:** Track attendance weekly")
    
    # Check backlogs
    if 'num_backlogs' in student_row.index:
        if student_row['num_backlogs'] > 0:
            recommendations.append("🎯 **Action:** Create personalized catch-up plan for backlogs")
    
    # Check marks trend
    if 'marks_trend' in student_row.index:
        if student_row['marks_trend'] < -5:
            recommendations.append("📚 **Support:** Arrange subject tutoring immediately")
        elif student_row['marks_trend'] < 0:
            recommendations.append("📖 **Watch:** Academic performance declining")
    
    # Check engagement
    if 'engagement_score' in student_row.index:
        if student_row['engagement_score'] < 50:
            recommendations.append("💬 **Intervene:** One-on-one mentoring session needed")
    
    # Check income
    if 'income_group' in student_row.index:
        if student_row['income_group'] == 'low':
            recommendations.append("💰 **Explore:** Financial aid and scholarship options")
    
    # Default if no specific recommendations
    if not recommendations:
        recommendations.append("✅ **Continue:** Maintain current support level")
        recommendations.append("👥 **Regular:** Monthly check-in with academic advisor")
    
    return recommendations

# ============================================
# SIDEBAR
# ============================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/student-center.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Go to",
        ["📊 Dashboard", "🔍 Student Search", "📤 Batch Upload", "📈 Analytics"]
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("""
    **Student Risk Prediction System**
    
    Early identification of at-risk students using machine learning.
    
    Built with:
    - Random Forest Classifier
    - SHAP Explanations
    - Streamlit Dashboard
    """)

# ============================================
# MAIN HEADER
# ============================================
st.markdown('<div class="main-title">🎓 Student Engagement Risk Dashboard</div>', unsafe_allow_html=True)
st.markdown("**AI-Powered Early Warning System for Academic Success**")
st.markdown("---")

# ============================================
# PAGE 1: DASHBOARD OVERVIEW
# ============================================
if page == "📊 Dashboard":
    
    # Key Metrics Row
    st.header("📊 Key Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    total_students = len(df)
    high_risk_count = df['risk_label'].sum() if 'risk_label' in df.columns else 0
    high_risk_pct = (high_risk_count / total_students * 100) if total_students > 0 else 0
    att_col = get_col(df, 'attendance_percent', 'attendance_rate')
    back_col = get_col(df, 'num_backlogs', 'Backlogs')
    marks_col = get_col(df, 'avg_internal_marks', 'cgpa')

    avg_attendance = df[att_col].mean() if att_col else 0
    avg_marks = df[marks_col].mean() if marks_col else 0
    
    col1.metric("👥 Total Students", f"{total_students:,}")
    col2.metric("⚠️ At Risk", f"{high_risk_count} ({high_risk_pct:.1f}%)")
    col3.metric("📅 Avg Attendance", f"{avg_attendance:.1f}%")
    col4.metric("📝 Avg Marks", f"{avg_marks:.1f}")
    
    st.markdown("---")
    
    # Risk Distribution Chart
    st.header("📊 Risk Distribution")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Predict risk for all students
        if 'risk_score' not in df.columns:
            with st.spinner("Calculating risk scores..."):
                drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out']
                drop_cols = [c for c in drop_cols if c in df.columns]

                X_all = df[[c for c in df.columns if c not in drop_cols]]
                X_all = pd.get_dummies(X_all, drop_first=True)
                X_all = X_all.reindex(columns=model_columns, fill_value=0)

                df['risk_score'] = model.predict_proba(X_all)[:, 1]
        
        # Categorize
        df['risk_category'] = df['risk_score'].apply(
            lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
        )
        
        risk_counts = df['risk_category'].value_counts()
        
        # Pie chart
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Students by Risk Level",
            color=risk_counts.index,
            color_discrete_map={'High': '#d62728', 'Medium': '#ff7f0e', 'Low': '#2ca02c'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_right:
        # Risk score distribution histogram
        fig_hist = px.histogram(
            df,
            x='risk_score',
            nbins=30,
            title="Risk Score Distribution",
            labels={'risk_score': 'Risk Score'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.add_vline(x=0.4, line_dash="dash", line_color="orange", annotation_text="Medium")
        fig_hist.add_vline(x=0.7, line_dash="dash", line_color="red", annotation_text="High")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("---")
    
    # Top At-Risk Students
    st.header("⚠️ Top 10 At-Risk Students (Immediate Action Needed)")

    # Build display columns dynamically to avoid KeyErrors when column names differ
    display_cols = ['student_id'] if 'student_id' in df.columns else []
    if 'name' in df.columns:
        display_cols.append('name')
    if att_col:
        display_cols.append(att_col)
    if back_col:
        display_cols.append(back_col)
    display_cols.append('risk_score')

    top_risk = df.nlargest(10, 'risk_score')[display_cols].copy()

    # Format values if present
    top_risk['risk_score'] = top_risk['risk_score'].apply(lambda x: f"{x:.2f}")
    if att_col:
        try:
            top_risk[att_col] = top_risk[att_col].apply(lambda x: f"{x:.1f}%")
        except Exception:
            pass

    # Build column config mapping
    column_config = {}
    if 'student_id' in display_cols:
        column_config['student_id'] = "Student ID"
    if 'name' in display_cols:
        column_config['name'] = "Name"
    if att_col:
        column_config[att_col] = "Attendance"
    if back_col:
        column_config[back_col] = "Backlogs"
    column_config['risk_score'] = "Risk Score"

    st.dataframe(
        top_risk,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )

# ============================================
# PAGE 2: STUDENT SEARCH
# ============================================
elif page == "🔍 Student Search":
    
    st.header("🔍 Individual Student Analysis")
    
    # Search/Select Student
    if 'student_id' in df.columns:
        # Create searchable options
        if 'name' in df.columns:
            options = [f"{sid} - {name}" for sid, name in zip(df['student_id'], df['name'])]
            search_by = st.selectbox(
                "🔎 Search by Student ID or Name",
                options,
                help="Type to search"
            )
            selected_id = search_by.split(" - ")[0]
        else:
            selected_id = st.selectbox("🔎 Select Student ID", df['student_id'].astype(str).tolist())
        
        student_row = df[df['student_id'].astype(str) == str(selected_id)].iloc[0]
        student_idx = df[df['student_id'].astype(str) == str(selected_id)].index[0]
    else:
        st.warning("No student_id column found. Showing first student.")
        student_row = df.iloc[0]
        student_idx = 0
    
    st.markdown("---")
    
    # Student Information Section
    col_info, col_risk = st.columns([1, 1])
    
    with col_info:
        st.subheader("📋 Student Information")
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.write(f"**ID:** {student_row.get('student_id', 'N/A')}")
            st.write(f"**Name:** {student_row.get('name', 'N/A')}")
            att_value = None
            if att_col:
                att_value = student_row.get(att_col, 0)
            else:
                att_value = student_row.get('attendance_percent', student_row.get('attendance_rate', 0))
            st.write(f"**Attendance:** {att_value:.1f}%")
        
        with info_col2:
            avg_marks_val = student_row.get('avg_internal_marks', student_row.get('cgpa', 0))
            st.write(f"**Avg Marks:** {avg_marks_val:.1f}")
            back_val = None
            if back_col:
                back_val = student_row.get(back_col, 0)
            else:
                back_val = student_row.get('num_backlogs', student_row.get('Backlogs', 0))
            st.write(f"**Backlogs:** {back_val}")
            st.write(f"**Marks Trend:** {student_row.get('marks_trend', 0):.1f}")
    
    with col_risk:
        st.subheader("⚠️ Risk Assessment")
        
        # Predict risk
        drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out']
        student_data = student_row.drop(labels=[c for c in drop_cols if c in student_row.index], errors='ignore')
        risk_score = predict_risk(student_data)
        
        risk_cat, risk_class, risk_color = get_risk_category(risk_score)
        
        # Risk gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Risk Score"},
            gauge={
                'axis': {'range': [None, 1]},
                'bar': {'color': risk_color},
                'steps': [
                    {'range': [0, 0.4], 'color': "lightgreen"},
                    {'range': [0.4, 0.7], 'color': "lightyellow"},
                    {'range': [0.7, 1], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.7
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
        st.markdown(f"### <span class='{risk_class}'>{risk_cat}</span>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    # SHAP Explanation Section
    st.header("🧠 Why This Prediction?")
    
    if shap_data:
        # If we have true SHAP values (old behavior)
        if isinstance(shap_data, dict) and 'shap_values' in shap_data:
            try:
                shap_vals = shap_data['shap_values'][student_idx]
                feature_names = shap_data['feature_names']

                explanation_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Impact': shap_vals
                }).sort_values('Impact', key=abs, ascending=False).head(10)

                explanation_df['Direction'] = explanation_df['Impact'].apply(
                    lambda x: '🔴 Increases Risk' if x > 0 else '🟢 Decreases Risk'
                )
                explanation_df['Impact_Abs'] = explanation_df['Impact'].abs().round(3)

                st.dataframe(
                    explanation_df[['Feature', 'Impact_Abs', 'Direction']],
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Feature": "Factor",
                        "Impact_Abs": "Impact Strength",
                        "Direction": "Effect"
                    }
                )

                fig_shap = px.bar(
                    explanation_df,
                    x='Impact',
                    y='Feature',
                    orientation='h',
                    title="Feature Impact on Prediction",
                    color='Impact',
                    color_continuous_scale=['green', 'white', 'red'],
                    color_continuous_midpoint=0
                )
                fig_shap.update_layout(height=400)
                st.plotly_chart(fig_shap, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying SHAP values: {e}")
        # Fallback: we loaded a permutation/feature-importance package
        elif isinstance(shap_data, dict) and 'feature_importance' in shap_data:
            try:
                fi = shap_data['feature_importance'] if 'feature_importance' in shap_data else None
                if fi is None:
                    # joblib dump returned plain DataFrame
                    fi = shap_data
                fi_top = fi.head(10)

                st.subheader("Top Features (Permutation Importance)")
                st.dataframe(fi_top, use_container_width=True, hide_index=True)

                fig_fi = px.bar(
                    fi_top,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top Features by Importance",
                    color='Importance',
                    color_continuous_scale='Blues'
                )
                fig_fi.update_layout(height=400)
                st.plotly_chart(fig_fi, use_container_width=True)
            except Exception as e:
                st.error(f"Error displaying feature importance: {e}")
        else:
            st.warning("⚠️ Explanations file found but format not recognized.")
    else:
        st.warning("⚠️ Explanations not available. Run `python src/compute_shap.py` or generate feature importance first.")
    
    st.markdown("---")
    
    # Recommendations Section
    st.header("💡 Recommended Interventions")
    
    recommendations = get_recommendations(student_row, shap_data)
    
    for rec in recommendations:
        st.markdown(f"- {rec}")

# ============================================
# PAGE 3: BATCH UPLOAD
# ============================================
elif page == "📤 Batch Upload":
    
    st.header("📤 Batch Prediction Upload")
    st.write("Upload a CSV file with student data to get risk predictions for multiple students at once.")
    
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV with the same columns as your training data"
    )
    
    if uploaded_file is not None:
        try:
            # Read uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Loaded {len(batch_df)} rows from uploaded file")
            
            st.subheader("📋 Preview of Uploaded Data")
            st.dataframe(batch_df.head(10), use_container_width=True)
            
            # Process predictions
            if st.button("🚀 Generate Predictions", type="primary"):
                with st.spinner("Calculating risk scores..."):
                    # Prepare data
                    drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out']
                    drop_cols = [c for c in drop_cols if c in batch_df.columns]
                    
                    # Keep ID columns for output
                    id_cols = batch_df[['student_id', 'name']] if 'student_id' in batch_df.columns and 'name' in batch_df.columns else batch_df[['student_id']] if 'student_id' in batch_df.columns else None
                    
                    X_batch = batch_df[[c for c in batch_df.columns if c not in drop_cols]]
                    X_batch = pd.get_dummies(X_batch, drop_first=True)
                    X_batch = X_batch.reindex(columns=model_columns, fill_value=0)
                    
                    # Predict
                    risk_scores = model.predict_proba(X_batch)[:, 1]
                    
                    # Create results dataframe
                    results_df = id_cols.copy() if id_cols is not None else pd.DataFrame()
                    results_df['risk_score'] = risk_scores
                    results_df['risk_category'] = results_df['risk_score'].apply(
                        lambda x: 'High' if x >= 0.7 else ('Medium' if x >= 0.4 else 'Low')
                    )
                    
                    st.success("✅ Predictions complete!")
                    
                    # Display results
                    st.subheader("📊 Prediction Results")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="student_risk_predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Summary stats
                    col1, col2, col3 = st.columns(3)
                    col1.metric("High Risk", f"{(results_df['risk_category']=='High').sum()}")
                    col2.metric("Medium Risk", f"{(results_df['risk_category']=='Medium').sum()}")
                    col3.metric("Low Risk", f"{(results_df['risk_category']=='Low').sum()}")
        
        except Exception as e:
            st.error(f"❌ Error processing file: {e}")
            st.info("Make sure your CSV has the same columns as the training data.")

# ============================================
# PAGE 4: ANALYTICS
# ============================================
elif page == "📈 Analytics":
    
    st.header("📈 Advanced Analytics")
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    st.write("Which factors matter most in predicting student risk?")
    
    model_pack = joblib.load("models/student_risk_model.pkl")
    feature_importance = model_pack['feature_importance'].head(15)
    
    fig_importance = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title="Top 15 Most Important Features",
        color='Importance',
        color_continuous_scale='Blues'
    )
    fig_importance.update_layout(height=500)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Analysis
    st.subheader("📊 Attendance vs Risk Score")
    
    if 'risk_score' not in df.columns:
        drop_cols = ['student_id', 'name', 'email', 'risk_label', 'dropped_out']
        drop_cols = [c for c in drop_cols if c in df.columns]
        X_all = df[[c for c in df.columns if c not in drop_cols]]
        X_all = pd.get_dummies(X_all, drop_first=True)
        X_all = X_all.reindex(columns=model_columns, fill_value=0)
        df['risk_score'] = model.predict_proba(X_all)[:, 1]
    
    if 'attendance_percent' in df.columns:
        fig_scatter = px.scatter(
            df,
            x='attendance_percent',
            y='risk_score',
            title="Relationship Between Attendance and Risk",
            labels={'attendance_percent': 'Attendance %', 'risk_score': 'Risk Score'},
            trendline="lowess",
            color='risk_score',
            color_continuous_scale='RdYlGn_r'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    
    # Model Performance
    st.subheader("🎯 Model Performance Metrics")
    
    metrics = model_pack['metrics']
    
    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("ROC AUC Score", f"{metrics['roc_auc']:.3f}")
    
    st.info("""
    **Model Performance Explanation:**
    - **Accuracy**: Overall correctness of predictions
    - **ROC AUC**: Ability to distinguish between at-risk and not-at-risk students (1.0 = perfect, 0.5 = random)
    """)

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>Student Risk Prediction Dashboard | Built with Streamlit & scikit-learn</p>
    <p>For questions or support, contact your data science team</p>
</div>
""", unsafe_allow_html=True)