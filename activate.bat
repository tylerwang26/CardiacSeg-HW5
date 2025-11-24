@echo off
set "PROJECT_ROOT=%~dp0"
set "PATH=%PROJECT_ROOT%.venv\Scripts;%PATH%"
set "PYTHONPATH=%PROJECT_ROOT%;%PYTHONPATH%"
set "nnUNet_raw=%PROJECT_ROOT%nnUNet_raw"
set "nnUNet_preprocessed=%PROJECT_ROOT%nnUNet_preprocessed"
set "nnUNet_results=%PROJECT_ROOT%nnUNet_results"

echo Environment activated!
where python
echo nnUNet_results: %nnUNet_results%