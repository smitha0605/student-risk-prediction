@echo off
echo Starting EduPulse Student Risk Prediction Dashboard...
cd /d "c:\Users\smith\Desktop\student-risk-prediction\app"
python -m streamlit run streamlit_app.py --server.port 8501 --server.headless false
pause
