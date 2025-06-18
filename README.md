# DrugSafeAI

DrugSafeAI is an advanced clinical decision support system that uses AI to predict medication risks, analyze drug interactions, and visualize potential side effects for healthcare professionals. The system incorporates machine learning, real-time data validation, user feedback mechanisms, and intelligent drug recommendations.

## Features

### Core Risk Assessment
- **Risk Prediction**: Assess medication safety risks based on patient demographics, medical history, and lab values
- **Drug Interaction Analysis**: Detect and display potential interactions between medications
- **Multi-factor Risk Assessment**: Consider age, weight, medical conditions, and lab values in risk calculations
- **Real-time Processing**: Fast prediction results using machine learning

### Advanced Analytics & Validation
- **Data Validation System**: Cross-reference drug data from multiple sources for accuracy and consistency
- **Distribution Analysis**: Analyze model prediction distributions to optimize risk thresholds
- **Evidence Quality Assessment**: Validate the quality and recency of medical evidence
- **Risk Factor Validation**: Ensure consistency in risk factor calculations across drug alternatives

### Intelligent Recommendations
- **Smart Drug Alternatives**: AI-powered recommendation system using machine learning and real-time data
- **Patient-Specific Recommendations**: Personalized suggestions based on individual patient profiles
- **Drug Similarity Analysis**: Calculate similarity between drugs based on multiple pharmacological factors
- **Interaction Severity Assessment**: Quantify the severity of drug-drug interactions

### User Feedback & Quality Assurance
- **Feedback System**: Collect and manage user feedback on drug information accuracy
- **Feedback Analytics**: Track feedback patterns and identify areas for improvement
- **Critical Issue Tracking**: Prioritize and manage critical feedback requiring immediate attention
- **Feedback Review Workflow**: Structured process for reviewing and addressing user feedback

### Interactive Visualizations
- **Risk Dashboard**: Displays overall risk levels using Chart.js
- **Drug Interaction Network**: Interactive network graph using vis.js
- **Anatomical Visualization**: Human anatomy visualization for body-system specific side effects
- **Distribution Analysis Charts**: Histograms and statistical plots for model performance analysis

### Data Management
- **Multi-Source Data Integration**: Combine data from OpenFDA API, FAERS, and local datasets
- **Automated Data Updates**: Regular updates from literature and adverse event reports
- **Data Quality Monitoring**: Continuous monitoring of data quality and consistency
- **Export Capabilities**: Generate comprehensive reports and data exports

## Installation

1. Clone the repository:
```bash
git clone https://github.com/mdoukan/DrugSafeAI.git
cd DrugSafeAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your OpenFDA API key:
```
OPENFDA_API_KEY=your_api_key_here
```

## Usage

### Basic Setup
1. Train the model:
```bash
python train_script.py
```

2. Start the server:
```bash
python run.py
```

3. Access the web interface at http://localhost:5000

### Advanced Features

#### Data Validation
```bash
python data_validation_system.py
```

#### Distribution Analysis
```bash
python analyze_distribution.py
```

#### Feedback System
```bash
python feedback_system.py
```

#### Intelligent Recommendations
```bash
python intelligent_drug_recommendation.py
```

## API Endpoints

### Predict Side Effects
```
POST /predict
Content-Type: application/json

{
    "age": 45,
    "weight": 70,
    "height": 170,
    "sex": 1,
    "medications": ["aspirin", "metformin"],
    "serious": 1,
    "medical_history": ["Hypertension", "Diabetes"]
}
```

### Get Side Effects Information
```
POST /api/side-effects
Content-Type: application/json

{
    "medications": ["aspirin", "metformin"]
}
```

### Submit Feedback
```
POST /api/feedback
Content-Type: application/json

{
    "drug_name": "aspirin",
    "feedback_type": "inaccuracy",
    "severity": "moderate",
    "description": "Risk factor seems too high for this patient population"
}
```

### Get Drug Recommendations
```
POST /api/recommendations
Content-Type: application/json

{
    "current_drug": "aspirin",
    "patient_data": {
        "age": 65,
        "medical_history": ["Hypertension"],
        "medications": ["lisinopril"]
    }
}
```

## Key Components

### Prediction Service
The prediction service combines multiple data sources to provide comprehensive risk analysis:
- Machine learning model predictions using XGBoost
- FDA adverse event data (FAERS) integration
- Drug interaction detection and severity assessment
- Evidence-based risk calculations with medical history adjustments

### Data Validation System
Ensures data quality and consistency across multiple sources:
- Cross-references drug data from different sources
- Validates risk factor ranges and consistency
- Checks evidence quality and recency
- Generates comprehensive validation reports

### Feedback Management System
Collects and manages user feedback for continuous improvement:
- SQLite database for feedback storage
- Categorization by severity and type
- Analytics dashboard for feedback patterns
- Review workflow for addressing issues

### Intelligent Recommendation Engine
AI-powered drug recommendation system:
- Machine learning-based similarity calculations
- Patient-specific risk scoring
- Multi-factor drug comparison
- Evidence-based alternative suggestions

## Model Analysis Tools

### Distribution Analysis
- Analyze model prediction distributions
- Optimize risk thresholds based on data
- Generate statistical visualizations
- Compare current vs. suggested thresholds

### Performance Monitoring
- Track model performance over time
- Monitor prediction accuracy
- Identify areas for improvement
- Generate performance reports

## Technologies Used

- **Backend**: Python 3.10+, Flask
- **Machine Learning**: XGBoost, scikit-learn, pandas, numpy
- **Data Visualization**: Chart.js, vis.js, matplotlib
- **Frontend**: Bootstrap 5, HTML5, CSS3, JavaScript
- **Database**: SQLite (for feedback system)
- **APIs**: OpenFDA API, FAERS (FDA Adverse Event Reporting System)
- **Data Analysis**: pandas, numpy, matplotlib, seaborn

## Project Structure

```
DrugSafeAI/
├── app/                     # Main application package
│   ├── __init__.py          # Flask application initialization
│   ├── routes.py            # API and web routes
│   ├── routes_user.py       # User-specific routes
│   ├── models/              # ML model definitions and utilities
│   ├── services/            # Core services for data retrieval and processing
│   ├── static/              # Static assets (CSS, JS, images)
│   └── templates/           # HTML templates
├── scripts/                 # Utility scripts for data processing
├── feedback_system.py       # User feedback management system
├── intelligent_drug_recommendation.py  # AI-powered drug recommendations
├── data_validation_system.py # Data quality validation system
├── analyze_distribution.py  # Model distribution analysis
├── test_model.py           # Model testing and validation
├── plot_histogram.py       # Visualization utilities
├── run.py                  # Application entry point
├── train_script.py         # Model training script
├── update_model_script.py  # Model updating with new data sources
├── config.py               # Configuration settings
└── requirements.txt        # Python dependencies
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

For technical support or questions about the system, please refer to the DEVELOPER.md file for detailed technical documentation. 