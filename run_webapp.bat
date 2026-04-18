@echo off
set SCRIPT_DIR=%~dp0
if "%DATAPATH%"=="" (
  set DATAPATH=%SCRIPT_DIR%dataset
)
python run_webapp.py
