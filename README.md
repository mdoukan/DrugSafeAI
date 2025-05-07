# DrugSafeAI

DrugSafeAI is an AI-powered drug safety prediction system that helps healthcare professionals and patients assess potential side effects and drug interactions.

## Features

- Predict potential side effects based on patient data and medications
- Analyze drug interactions
- Real-time predictions using machine learning
- RESTful API for easy integration
- User-friendly web interface

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
    "serious": 1
}
```

## Technologies Used

- Python 3.10+
- Flask
- XGBoost
- scikit-learn
- pandas
- OpenFDA API

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request 