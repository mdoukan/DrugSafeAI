from flask import Blueprint, render_template, request, jsonify
import logging
import traceback
from app.services.prediction_service import PredictionService
from app.services.faers_service import FaersService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)
prediction_service = PredictionService()
faers_service = FaersService()

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/api/side-effects', methods=['POST'])
def get_side_effects():
    try:
        data = request.get_json()
        logger.info(f"Received side effects request with data: {data}")
        
        if not data or 'medications' not in data:
            logger.error("No medications data received")
            return jsonify({'error': 'No medications provided'}), 400
            
        medications = data['medications']
        if not isinstance(medications, list):
            logger.error("Medications must be a list")
            return jsonify({'error': 'Medications must be a list'}), 400
            
        # Her ilaç için yan etkileri al
        results = {}
        for medication in medications:
            try:
                # FAERS servisinden yan etkileri al
                side_effects = faers_service.get_drug_side_effects(medication)
                
                # Eğer yan etki yoksa veya hata oluştuysa, örnek veri kullan
                if not side_effects:
                    results[medication] = generate_sample_side_effects(medication)
                else:
                    results[medication] = side_effects
                    
            except Exception as e:
                logger.error(f"Error fetching side effects for {medication}: {str(e)}")
                results[medication] = generate_sample_side_effects(medication)
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in side effects API: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

def generate_sample_side_effects(medication):
    """Örnek yan etki verileri oluştur"""
    sample_data = {
        'aspirin': [
            {'name': 'Stomach Upset', 'probability': 0.45, 'severity': 'Medium', 
             'description': 'Stomach irritation or discomfort', 
             'recommendation': 'Take with food to reduce stomach upset'},
            {'name': 'Bleeding', 'probability': 0.30, 'severity': 'High', 
             'description': 'Increased risk of bleeding due to blood thinning properties', 
             'recommendation': 'Stop medication and consult doctor if unusual bleeding occurs'},
            {'name': 'Heartburn', 'probability': 0.25, 'severity': 'Low', 
             'description': 'Burning sensation in chest', 
             'recommendation': 'Take with plenty of water'}
        ],
        'ibuprofen': [
            {'name': 'Stomach Pain', 'probability': 0.40, 'severity': 'Medium', 
             'description': 'Pain or discomfort in stomach', 
             'recommendation': 'Take with food to reduce stomach irritation'},
            {'name': 'Dizziness', 'probability': 0.20, 'severity': 'Medium', 
             'description': 'Feeling lightheaded or unsteady', 
             'recommendation': 'Avoid driving or operating machinery if affected'},
            {'name': 'Headache', 'probability': 0.15, 'severity': 'Low', 
             'description': 'Pain in the head or temples', 
             'recommendation': 'Usually temporary; consult doctor if persistent'}
        ],
        'amoxicillin': [
            {'name': 'Diarrhea', 'probability': 0.35, 'severity': 'Medium', 
             'description': 'Loose, watery stools', 
             'recommendation': 'Stay hydrated and consult doctor if severe'},
            {'name': 'Rash', 'probability': 0.25, 'severity': 'Medium', 
             'description': 'Skin eruption or discoloration', 
             'recommendation': 'Stop medication and contact doctor immediately'},
            {'name': 'Nausea', 'probability': 0.20, 'severity': 'Low', 
             'description': 'Feeling of sickness with an inclination to vomit', 
             'recommendation': 'Take with food to reduce nausea'}
        ]
    }
    
    # İlaç için örnek veri varsa onu döndür, yoksa varsayılan veri oluştur
    if medication in sample_data:
        return sample_data[medication]
    else:
        return [
            {'name': 'General Side Effect 1', 'probability': 0.30, 'severity': 'Medium', 
             'description': f'Common side effect of {medication}', 
             'recommendation': 'Monitor and consult doctor if persistent'},
            {'name': 'General Side Effect 2', 'probability': 0.20, 'severity': 'Low', 
             'description': f'Less common reaction to {medication}', 
             'recommendation': 'Usually resolves without intervention'},
            {'name': 'General Side Effect 3', 'probability': 0.10, 'severity': 'Low', 
             'description': f'Rare reaction to {medication}', 
             'recommendation': 'No specific action required unless severe'}
        ]

@main.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logger.info(f"Received prediction request with data: {data}")
        
        if not data:
            logger.error("No data received in request")
            return jsonify({'error': 'No data provided'}), 400
            
        # Validate required fields
        required_fields = ['age', 'weight', 'height', 'sex', 'medications', 'serious']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({'error': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
        # Validate data types
        try:
            data['age'] = float(data['age'])
            data['weight'] = float(data['weight'])
            data['height'] = float(data['height'])
            data['sex'] = int(data['sex'])
            data['serious'] = int(data['serious'])
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid data type: {str(e)}")
            return jsonify({'error': 'Invalid data type for numeric fields'}), 400
            
        # Validate medications
        if not isinstance(data.get('medications', []), list):
            logger.error("Medications must be a list")
            return jsonify({'error': 'Medications must be a list'}), 400
            
        # Validate optional fields if present
        if 'medical_history' in data and not isinstance(data['medical_history'], list):
            logger.error("Medical history must be a list")
            return jsonify({'error': 'Medical history must be a list'}), 400
            
        if 'laboratory_tests' in data and not isinstance(data['laboratory_tests'], list):
            logger.error("Laboratory tests must be a list")
            return jsonify({'error': 'Laboratory tests must be a list'}), 400
            
        # Make prediction
        try:
            results = prediction_service.predict_side_effects(data)
            logger.info(f"Prediction results: {results}")
            return jsonify(results)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'error': f'Prediction error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500 