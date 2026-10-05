@echo off
title MetaTrader 5 Algorithmic Trading App
color 0A
echo ============================================================
echo      METATRADER 5 ALGORITHMIC TRADING TERMINAL
echo ============================================================
echo.
echo Checking dependencies...
python -m pip install -r requirements.txt --quiet
echo Starting Streamlit Web Terminal...
echo.
echo Open your web browser at the URL shown below (typically http://localhost:8501)
echo.
python -m streamlit run app.py
pause
