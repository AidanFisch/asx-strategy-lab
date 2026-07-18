@echo off
REM ---------------------------------------------------------------------------
REM WEEKLY rescan (heavy, ~1-2h). Refresh all data, re-run the full v2 scan,
REM rebuild the trading plans (with walk-forward), and regenerate the dashboard.
REM Run this on a weekend; the daily monitor (run_scan.cmd) uses its output.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
if not exist logs mkdir logs
echo ============================================================ >> logs\rescan.log
echo Rescan started %DATE% %TIME% >> logs\rescan.log
py download_data.py --interval 1d --universe          >> logs\rescan.log 2>&1
py download_data.py --interval 1d --universe-file data/universe_asia.csv >> logs\rescan.log 2>&1
py -m backtest.scanner2 --interval 1d                  >> logs\rescan.log 2>&1
py -m plans --interval 1d                              >> logs\rescan.log 2>&1
py -m robustness --interval 1d                         >> logs\rescan.log 2>&1
py -m results.dashboard2 --interval 1d --pages         >> logs\rescan.log 2>&1
echo Rescan finished %DATE% %TIME% >> logs\rescan.log
