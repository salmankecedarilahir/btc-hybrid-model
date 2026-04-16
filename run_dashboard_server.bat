@echo off
echo ========================================
echo   BTC Hybrid Model - Dashboard Server
echo ========================================
echo.
echo Starting HTTP Server on port 8000...
echo Dashboard: http://localhost:8000/dashboard.html
echo.
echo Press Ctrl+C to stop
echo.

python -m http.server 8000
