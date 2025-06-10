#!/bin/bash

# DrugSafeAI Setup Script

echo "Setting up DrugSafeAI environment..."

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
if [[ $(echo "$python_version" | cut -d. -f1) -lt 3 ]] || [[ $(echo "$python_version" | cut -d. -f2) -lt 8 ]]; then
    echo "Python version $python_version detected. DrugSafeAI requires Python 3.8 or higher."
    exit 1
fi

# Create and activate virtual environment
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    echo "OPENFDA_API_KEY=your_api_key_here" > .env
    echo "Please edit the .env file and add your OpenFDA API key"
fi

# Train the model
echo "Would you like to train the model now? (y/n)"
read train_model

if [[ "$train_model" == "y" || "$train_model" == "Y" ]]; then
    echo "Training model..."
    python train_script.py
else
    echo "You can train the model later by running: python train_script.py"
fi

echo "Setup complete! To start the server, run: python run.py"
echo "Then access the web interface at http://localhost:5000" 