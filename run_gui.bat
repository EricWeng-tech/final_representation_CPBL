@echo off
setlocal
cd /d "%~dp0"
python scripts\gui_predict.py
endlocal
