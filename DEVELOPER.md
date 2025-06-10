# DrugSafeAI Developer Guide

This document provides technical information for developers interested in contributing to or extending DrugSafeAI.

## Project Structure

```
DrugSafeAI/
├── app/                     # Main application package
│   ├── __init__.py          # Flask application initialization
│   ├── routes.py            # API and web routes
│   ├── models/              # ML model definitions and utilities
│   ├── services/            # Core services for data retrieval and processing
│   ├── static/              # Static assets (CSS, JS, images)
│   └── templates/           # HTML templates
├── scripts/                 # Utility scripts for data processing
├── run.py                   # Application entry point
├── train_script.py          # Model training script
├── update_model_script.py   # Model updating with new data sources
├── config.py                # Configuration settings
└── requirements.txt         # Python dependencies
```

## Key Components

### Prediction Service

The prediction service combines multiple data sources to provide risk analysis:
- Machine learning model predictions
- FDA adverse event data (FAERS)
- Drug interaction detection
- Evidence-based risk calculations

### Data Pipeline

1. **Data Collection**: Medication data is collected from OpenFDA API and local datasets
2. **Preprocessing**: Data is cleaned and transformed for model input
3. **Model Training**: XGBoost model trained to predict risks
4. **Regular Updates**: Automatic updates from literature and adverse event reports

### Visualization System

1. **Risk Dashboard**: Displays overall risk levels using Chart.js
2. **Drug Interaction Network**: Interactive network graph using vis.js
3. **Anatomical Visualization**: Human body SVG map with interactive hotspots

## Adding New Features

### Adding a New Data Source

1. Create a new service in `app/services/`
2. Implement the data retrieval and parsing functions
3. Register the service in the appropriate prediction pipeline
4. Update the model training script to incorporate the new data source

### Adding a New Visualization

1. Create HTML structure in the appropriate template
2. Add JavaScript code in a dedicated file in `static/js/`
3. Ensure the route provides the necessary data
4. Update the CSS styling as needed

### Extending the ML Model

1. Modify the feature engineering in `train_script.py`
2. Adjust the model parameters or architecture in `app/models/`
3. Run validation tests to ensure performance
4. Update the prediction pipeline to use the new model

## Testing

We use pytest for unit and integration testing. Run tests with:

```bash
python -m pytest
```

Key test files:
- `test_model.py`: Tests for the ML model functionality
- `test_predictions.py`: Tests for the prediction pipeline

## API Documentation

### Predict Side Effects

```
POST /predict
```

Parameters:
- `age` (int): Patient age
- `weight` (float): Weight in kg
- `height` (float): Height in cm
- `sex` (int): 0 for female, 1 for male
- `medications` (list): List of medication names
- `serious` (int): Flag for serious conditions (0/1)
- `medical_history` (list, optional): List of medical conditions

### Drug Interactions

```
POST /drug-interactions
```

Parameters:
- `medications` (list): List of medication names

## Deployment

The application can be deployed using various methods:

1. **Docker**:
   Build the Docker image using the provided Dockerfile and deploy to any Docker-compatible environment.

2. **Traditional Server**:
   Use Gunicorn or uWSGI with Nginx for production deployment.

3. **Cloud Platforms**:
   Compatible with AWS, Azure, GCP with minimal configuration changes.

## Contribution Guidelines

1. Create a feature branch from `main`
2. Implement changes with appropriate tests
3. Ensure all tests pass and documentation is updated
4. Submit a pull request with a clear description of changes 