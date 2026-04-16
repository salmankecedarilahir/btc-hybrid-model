@echo off
REM ╔══════════════════════════════════════════════════════════════╗
REM ║   run_pipeline.bat — BTC Hybrid Model V7                    ║
REM ║   Urutan WAJIB: setiap step harus selesai sebelum lanjut    ║
REM ╚══════════════════════════════════════════════════════════════╝

echo.
echo ══════════════════════════════════════════════════════════════
echo   BTC HYBRID MODEL V7 — FULL PIPELINE
echo ══════════════════════════════════════════════════════════════
echo.

echo [STEP 1/13] Fetching market data...
python data_fetcher.py
if errorlevel 1 ( echo [ERROR] data_fetcher.py GAGAL & pause & exit /b 1 )
echo [OK] Step 1 selesai & echo.

echo [STEP 2/13] Cleaning data...
python data_cleaner.py
if errorlevel 1 ( echo [ERROR] data_cleaner.py GAGAL & pause & exit /b 1 )
echo [OK] Step 2 selesai & echo.

echo [STEP 3/13] Computing indicators...
python indicators.py
if errorlevel 1 ( echo [ERROR] indicators.py GAGAL & pause & exit /b 1 )
echo [OK] Step 3 selesai & echo.

echo [STEP 4/13] Detecting market regimes...
python regime_engine.py
if errorlevel 1 ( echo [ERROR] regime_engine.py GAGAL & pause & exit /b 1 )
echo [OK] Step 4 selesai & echo.

echo [STEP 5/13] Fetching derivatives data...
python derivatives_fetcher.py
if errorlevel 1 ( echo [ERROR] derivatives_fetcher.py GAGAL & pause & exit /b 1 )
echo [OK] Step 5 selesai & echo.

echo [STEP 6/13] Processing derivatives...
python derivatives_engine.py
if errorlevel 1 ( echo [ERROR] derivatives_engine.py GAGAL & pause & exit /b 1 )
echo [OK] Step 6 selesai & echo.

echo [STEP 7/13] Generating hybrid signals...
python hybrid_engine.py
if errorlevel 1 ( echo [ERROR] hybrid_engine.py GAGAL & pause & exit /b 1 )
echo [OK] Step 7 selesai & echo.

echo [STEP 8/13] Enhancing signal quality V7...
python signal_enhancer_v7.py --report
if errorlevel 1 ( echo [ERROR] signal_enhancer_v7.py GAGAL & pause & exit /b 1 )
echo [OK] Step 8 selesai & echo.

echo [STEP 9/13] Running backtest engine...
python backtest_engine.py
if errorlevel 1 ( echo [ERROR] backtest_engine.py GAGAL & pause & exit /b 1 )
echo [OK] Step 9 selesai & echo.

echo [STEP 10/13] Applying risk engine V6...
python risk_engine_v6.py
if errorlevel 1 ( echo [ERROR] risk_engine_v6.py GAGAL & pause & exit /b 1 )
echo [OK] Step 10 selesai & echo.

echo [STEP 11/13] Computing live signal...
python live_signal_runner.py
if errorlevel 1 ( echo [ERROR] live_signal_runner.py GAGAL & pause & exit /b 1 )
echo [OK] Step 11 selesai & echo.

echo [STEP 12/13] Updating paper trader...
REM Hapus log corrupt jika ada (paper trader akan buat ulang)
if exist data\paper_trading_log.csv (
    python -c "import pandas as pd; pd.read_csv('data/paper_trading_log.csv')" 2>nul || del data\paper_trading_log.csv
)
python paper_trader.py --once
if errorlevel 1 ( echo [ERROR] paper_trader.py GAGAL & pause & exit /b 1 )
echo [OK] Step 12 selesai & echo.

echo [STEP 13/13] Sending Telegram notification...
python telegram_notifier.py
if errorlevel 1 ( echo [WARNING] telegram_notifier.py GAGAL - cek token/koneksi & echo. )
echo [OK] Step 13 selesai & echo.

echo ══════════════════════════════════════════════════════════════
echo   PIPELINE SELESAI — Notifikasi Telegram terkirim
echo.
echo   Lanjutkan dengan:
echo     python monte_carlo_simulation.py -n 10000
echo     python walk_forward_test.py --detail
echo     python audit_final_comprehensive.py
echo ══════════════════════════════════════════════════════════════
echo.
pause
