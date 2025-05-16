@echo off
:: DrugSafeAI Setup Script for Windows

echo Setting up DrugSafeAI environment...

:: Check if Python is installed
python --version 2>NUL
if errorlevel 1 (
    echo Python not found. Please install Python 3.8 or higher.
    exit /b 1
)

:: Create and activate virtual environment
echo Creating virtual environment...
python -m venv venv

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

:: Create .env file if it doesn't exist
if not exist .env (
    echo Creating .env file...
    echo OPENFDA_API_KEY=your_api_key_here > .env
    echo Please edit the .env file and add your OpenFDA API key
)

:: Train the model
set /p train_model="Would you like to train the model now? (y/n): "
if /i "%train_model%"=="y" (
    echo Training model...
    python train_script.py
) else (
    echo You can train the model later by running: python train_script.py
)

echo Setup complete! To start the server, run: python run.py
echo Then access the web interface at http://localhost:5000

pause 