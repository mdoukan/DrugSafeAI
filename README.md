# DrugSafeAI

DrugSafeAI is an advanced clinical decision support system that uses AI to predict medication risks, analyze drug interactions, and visualize potential side effects for healthcare professionals.

## Features

- **Risk Prediction**: Assess medication safety risks based on patient demographics, medical history, and lab values
- **Drug Interaction Analysis**: Detect and display potential interactions between medications
- **Interactive Visualizations**:
  - Risk level gauge and comparative medication risk charts
  - Drug interaction network visualization
  - Human anatomy visualization for body-system specific side effects
- **Evidence-Based Recommendations**: Get alternative medication suggestions based on risk analysis
- **Real-time Processing**: Fast prediction results using machine learning
- **Multi-factor Risk Assessment**: Consider age, weight, medical conditions, and lab values in risk calculations

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

1. Train the model:
```bash
python train_script.py
```

2. Start the server:
```bash
python run.py
```

3. Access the web interface at http://localhost:5000

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

## Key Visualizations

### Drug Interaction Network
The application visualizes medication interactions through an interactive network diagram, displaying potential interactions between multiple medications and their severity levels.

### Anatomical Side Effect Visualization
DrugSafeAI includes an interactive human anatomy visualization that highlights body systems affected by potential side effects, providing an intuitive way for healthcare providers to communicate risks to patients.

## Technologies Used

- Python 3.10+
- Flask
- XGBoost
- scikit-learn
- pandas
- Chart.js for data visualization
- vis.js for network visualization
- Bootstrap 5 for UI
- OpenFDA API for medication data
- FAERS (FDA Adverse Event Reporting System) for evidence-based risk analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 