@echo off
REM ---------------------------------------------------------------------------
REM DAILY monitor (run after ASX close). Fast: refreshes only the plan tickers'
REM prices, then reports BUY / SELL / HOLD and sends the Telegram summary.
REM Telegram creds are read from the environment. Logs to logs\monitor.log.
REM ---------------------------------------------------------------------------
cd /d "%~dp0"
if not exist logs mkdir logs
echo ============================================================ >> logs\monitor.log
echo Run started %DATE% %TIME% >> logs\monitor.log
py -m live.monitor --interval 1d --refresh >> logs\monitor.log 2>&1
echo Run finished %DATE% %TIME% >> logs\monitor.log
