@echo off
set PYTHONPATH=E:\DEV-PROJECT\01-open-vocabulary-object-finder\repo
set STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
cd /d E:\DEV-PROJECT\01-open-vocabulary-object-finder\repo
"C:\Users\Trey\anaconda3\_conda.exe" run -n ovod python -m streamlit run demo_app.py --server.port 8501 --server.headless true