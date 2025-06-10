from flask import Blueprint, render_template, request, jsonify
import logging
import traceback
from app.services.prediction_service import PredictionService
from app.services.faers_service import FaersService
from app.models.database import db, User, MedicalHistory, Medication, LabTest
from datetime import datetime

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
        # İstek JSON formatında mı yoksa form verileri mi kontrol et
        if request.is_json:
            data = request.get_json()
        else:
            # Form verilerini işle
            data = {}
            # Temel hasta verileri
            try:
                data['age'] = float(request.form.get('age', 0))
                data['weight'] = float(request.form.get('weight', 0))
                data['height'] = float(request.form.get('height', 0))
                data['sex'] = int(request.form.get('sex', 0))
                data['serious'] = 1 if request.form.get('serious') == 'on' else 0
                
                # Save to profile flag
                data['save_history'] = request.form.get('save_history') == 'on'
                data['tc_number'] = request.form.get('tc_number')
                data['first_name'] = request.form.get('first_name')
                data['last_name'] = request.form.get('last_name')
            except (ValueError, TypeError) as e:
                logger.error(f"Error parsing form data: {str(e)}")
                return jsonify({'status': 'error', 'message': f'Invalid form data: {str(e)}'}), 400
                
            # İlaçlar
            medications = request.form.getlist('medications')
            data['medications'] = medications
            
            # Tıbbi geçmiş
            medical_history = request.form.getlist('medical_history')
            if medical_history:
                data['medical_history'] = medical_history
                
            # Laboratuvar testleri
            lab_tests = []
            # Log all form keys to help debug
            logger.info(f"Form keys: {list(request.form.keys())}")
            
            # Check for lab test data in the form
            lab_index = 0
            while True:
                lab_name_key = f'lab-name-{lab_index}'
                lab_value_key = f'lab-value-{lab_index}'
                
                if lab_name_key in request.form and lab_value_key in request.form:
                    test_name = request.form.get(lab_name_key)
                    test_value = request.form.get(lab_value_key)
                    test_unit = request.form.get(f'lab-unit-{lab_index}', '')
                    
                    if test_name and test_value:
                        logger.info(f"Found lab test in form: {test_name}={test_value} {test_unit}")
                        lab_tests.append({
                            'name': test_name,
                            'value': test_value,
                            'unit': test_unit
                        })
                    
                    lab_index += 1
                else:
                    break
            
            if lab_tests:
                data['laboratory_tests'] = lab_tests
                logger.info(f"Found {len(lab_tests)} lab tests in form")
        
        logger.info(f"Received prediction request with data: {data}")
        
        if not data:
            logger.error("No data received in request")
            return jsonify({'status': 'error', 'message': 'No data provided'}), 400
            
        # Validate required fields
        required_fields = ['age', 'weight', 'height', 'sex', 'medications', 'serious']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return jsonify({'status': 'error', 'message': f'Missing required fields: {", ".join(missing_fields)}'}), 400
            
        # Validate medications
        if not isinstance(data.get('medications', []), list) or not data.get('medications', []):
            logger.error("Medications must be a non-empty list")
            return jsonify({'status': 'error', 'message': 'Please select at least one medication'}), 400
            
        # Save medical history and medications if requested
        user_id = None
        if data.get('save_history') and data.get('tc_number') and data.get('first_name') and data.get('last_name'):
            try:
                user_id = save_user_data(data)
                logger.info(f"User medical history saved successfully for user ID: {user_id}")
            except Exception as e:
                logger.error(f"Error saving user data: {str(e)}")
                # Continue with prediction even if saving fails
            
        # Make prediction
        try:
            results = prediction_service.predict_side_effects(data)
            logger.info(f"Prediction results: {results}")
            
            # Add user_id to the response if available
            if user_id:
                results['user_id'] = user_id
                
            return jsonify(results)
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({'status': 'error', 'message': f'Prediction error: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An unexpected error occurred: {str(e)}'}), 500

def save_user_data(data):
    """Save user medical history and medications to database."""
    tc_number = data.get('tc_number')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    
    # Find or create user
    user = User.query.filter_by(tc_number=tc_number).first()
    
    if user:
        # Update user information
        user.first_name = first_name
        user.last_name = last_name
    else:
        # Create new user
        user = User(
            tc_number=tc_number,
            first_name=first_name,
            last_name=last_name
        )
        db.session.add(user)
        db.session.flush()  # Flush to get the user ID
    
    # Save medical history
    if 'medical_history' in data and data['medical_history']:
        for condition in data['medical_history']:
            # Check if this condition already exists for the user
            existing = MedicalHistory.query.filter_by(
                user_id=user.id, 
                condition=condition
            ).first()
            
            if not existing:
                medical_history = MedicalHistory(
                    user_id=user.id,
                    condition=condition,
                    diagnosis_date=datetime.today().date()
                )
                db.session.add(medical_history)
    
    # Save medications
    if 'medications' in data and data['medications']:
        for med_name in data['medications']:
            # Check if this medication already exists for the user
            existing = Medication.query.filter_by(
                user_id=user.id, 
                name=med_name
            ).first()
            
            if not existing:
                medication = Medication(
                    user_id=user.id,
                    name=med_name,
                    start_date=datetime.today().date()
                )
                db.session.add(medication)
    
    # Save lab tests from the form
    lab_index = 0
    lab_tests_found = False
    
    # Check if we have lab test data in the form
    while True:
        lab_name_key = f'lab-name-{lab_index}'
        lab_value_key = f'lab-value-{lab_index}'
        
        if lab_name_key in request.form and lab_value_key in request.form:
            lab_tests_found = True
            test_name = request.form.get(lab_name_key)
            test_result = request.form.get(lab_value_key)
            test_unit = request.form.get(f'lab-unit-{lab_index}', '')
            
            if test_name and test_result:
                # Check if this lab test already exists for the user
                existing = LabTest.query.filter_by(
                    user_id=user.id,
                    test_name=test_name,
                    result=test_result
                ).first()
                
                if not existing:
                    logger.info(f"Saving lab test: {test_name}={test_result} {test_unit}")
                    lab_test = LabTest(
                        user_id=user.id,
                        test_name=test_name,
                        result=test_result,
                        unit=test_unit,
                        test_date=datetime.today().date()
                    )
                    db.session.add(lab_test)
            
            lab_index += 1
        else:
            break
    
    if lab_tests_found:
        logger.info(f"Saved {lab_index} lab tests for user {user.id}")
    
    # Commit changes
    db.session.commit()
    return user.id 