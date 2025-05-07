from flask import Blueprint, render_template, request, jsonify
import logging
import traceback
from app.services.prediction_service import PredictionService

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

main = Blueprint('main', __name__)
prediction_service = PredictionService()

@main.route('/')
def index():
    return render_template('index.html')

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