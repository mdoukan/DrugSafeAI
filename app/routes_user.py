from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
import logging
import traceback
from app.models.database import db, User, MedicalHistory, Medication, LabTest
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

user_bp = Blueprint('user', __name__)

@user_bp.route('/user-home')
def user_home():
    """Render the user home page with the user form."""
    return render_template('user_home.html')

@user_bp.route('/user/find', methods=['POST'])
def find_user():
    """Find a user by TC number, first name, and last name."""
    try:
        tc_number = request.form.get('tc_number')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        logger.info(f"Finding user with TC: {tc_number}, Name: {first_name} {last_name}")
        
        # Validate input
        if not tc_number or not first_name or not last_name:
            logger.error("Missing required fields")
            return jsonify({'error': 'All fields are required'}), 400
            
        # Check if user exists
        user = User.query.filter_by(
            tc_number=tc_number, 
            first_name=first_name, 
            last_name=last_name
        ).first()
        
        if user:
            logger.info(f"User found: {user.id}")
            return jsonify({
                'status': 'success',
                'user_exists': True,
                'user_id': user.id,
                'message': 'User found successfully'
            })
        else:
            logger.info("User not found")
            return jsonify({
                'status': 'success',
                'user_exists': False,
                'message': 'User not found'
            })
            
    except Exception as e:
        logger.error(f"Error finding user: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@user_bp.route('/user/register', methods=['POST'])
def register_user():
    """Register a new user or update an existing user."""
    try:
        tc_number = request.form.get('tc_number')
        first_name = request.form.get('first_name')
        last_name = request.form.get('last_name')
        
        logger.info(f"Registering user with TC: {tc_number}, Name: {first_name} {last_name}")
        
        # Validate input
        if not tc_number or not first_name or not last_name:
            logger.error("Missing required fields")
            return jsonify({'error': 'All fields are required'}), 400
            
        # Check if user exists
        user = User.query.filter_by(tc_number=tc_number).first()
        
        if user:
            # Update user information
            user.first_name = first_name
            user.last_name = last_name
            logger.info(f"Updating existing user: {user.id}")
        else:
            # Create new user
            user = User(
                tc_number=tc_number,
                first_name=first_name,
                last_name=last_name
            )
            db.session.add(user)
            logger.info("Creating new user")
            
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'user_id': user.id,
            'message': 'User registered successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error registering user: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@user_bp.route('/user/<int:user_id>/medical-history', methods=['GET'])
def get_medical_history(user_id):
    """Get medical history, medications, and lab tests for a user."""
    try:
        user = User.query.get(user_id)
        
        if not user:
            logger.error(f"User not found: {user_id}")
            return jsonify({'error': 'User not found'}), 404
            
        # Get medical history
        medical_histories = MedicalHistory.query.filter_by(user_id=user_id).all()
        medical_history_list = [{
            'id': h.id,
            'condition': h.condition,
            'diagnosis_date': h.diagnosis_date.strftime('%Y-%m-%d') if h.diagnosis_date else None,
            'notes': h.notes
        } for h in medical_histories]
        
        # Get medications
        medications = Medication.query.filter_by(user_id=user_id).all()
        medication_list = [{
            'id': m.id,
            'name': m.name,
            'dosage': m.dosage,
            'frequency': m.frequency,
            'start_date': m.start_date.strftime('%Y-%m-%d') if m.start_date else None,
            'end_date': m.end_date.strftime('%Y-%m-%d') if m.end_date else None,
            'notes': m.notes
        } for m in medications]
        
        # Get lab tests
        lab_tests = LabTest.query.filter_by(user_id=user_id).all()
        lab_test_list = [{
            'id': t.id,
            'test_name': t.test_name,
            'result': t.result,
            'unit': t.unit,
            'test_date': t.test_date.strftime('%Y-%m-%d') if t.test_date else None,
            'notes': t.notes
        } for t in lab_tests]
        
        return jsonify({
            'status': 'success',
            'user': {
                'id': user.id,
                'tc_number': user.tc_number,
                'first_name': user.first_name,
                'last_name': user.last_name
            },
            'medical_history': medical_history_list,
            'medications': medication_list,
            'lab_tests': lab_test_list
        })
        
    except Exception as e:
        logger.error(f"Error getting medical history: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@user_bp.route('/user/<int:user_id>/medical-history', methods=['POST'])
def add_medical_history(user_id):
    """Add a new medical history record for a user."""
    try:
        user = User.query.get(user_id)
        
        if not user:
            logger.error(f"User not found: {user_id}")
            return jsonify({'error': 'User not found'}), 404
            
        condition = request.form.get('condition')
        diagnosis_date = request.form.get('diagnosis_date')
        notes = request.form.get('notes')
        
        # Validate input
        if not condition:
            logger.error("Missing required field: condition")
            return jsonify({'error': 'Condition is required'}), 400
            
        # Parse date if provided
        parsed_date = None
        if diagnosis_date:
            try:
                parsed_date = datetime.strptime(diagnosis_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error(f"Invalid date format: {diagnosis_date}")
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
                
        # Create new medical history record
        medical_history = MedicalHistory(
            user_id=user_id,
            condition=condition,
            diagnosis_date=parsed_date,
            notes=notes
        )
        
        db.session.add(medical_history)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'medical_history_id': medical_history.id,
            'message': 'Medical history added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding medical history: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@user_bp.route('/user/<int:user_id>/medication', methods=['POST'])
def add_medication(user_id):
    """Add a new medication record for a user."""
    try:
        user = User.query.get(user_id)
        
        if not user:
            logger.error(f"User not found: {user_id}")
            return jsonify({'error': 'User not found'}), 404
            
        name = request.form.get('name')
        dosage = request.form.get('dosage')
        frequency = request.form.get('frequency')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')
        notes = request.form.get('notes')
        
        # Validate input
        if not name:
            logger.error("Missing required field: name")
            return jsonify({'error': 'Medication name is required'}), 400
            
        # Parse dates if provided
        parsed_start_date = None
        if start_date:
            try:
                parsed_start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error(f"Invalid start date format: {start_date}")
                return jsonify({'error': 'Invalid start date format. Use YYYY-MM-DD'}), 400
                
        parsed_end_date = None
        if end_date:
            try:
                parsed_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error(f"Invalid end date format: {end_date}")
                return jsonify({'error': 'Invalid end date format. Use YYYY-MM-DD'}), 400
                
        # Create new medication record
        medication = Medication(
            user_id=user_id,
            name=name,
            dosage=dosage,
            frequency=frequency,
            start_date=parsed_start_date,
            end_date=parsed_end_date,
            notes=notes
        )
        
        db.session.add(medication)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'medication_id': medication.id,
            'message': 'Medication added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding medication: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500

@user_bp.route('/user/<int:user_id>/lab-test', methods=['POST'])
def add_lab_test(user_id):
    """Add a new lab test record for a user."""
    try:
        user = User.query.get(user_id)
        
        if not user:
            logger.error(f"User not found: {user_id}")
            return jsonify({'error': 'User not found'}), 404
            
        test_name = request.form.get('test_name')
        result = request.form.get('result')
        unit = request.form.get('unit')
        test_date = request.form.get('test_date')
        notes = request.form.get('notes')
        
        # Validate input
        if not test_name:
            logger.error("Missing required field: test_name")
            return jsonify({'error': 'Test name is required'}), 400
            
        if not result:
            logger.error("Missing required field: result")
            return jsonify({'error': 'Test result is required'}), 400
            
        # Parse date if provided
        parsed_test_date = None
        if test_date:
            try:
                parsed_test_date = datetime.strptime(test_date, '%Y-%m-%d').date()
            except ValueError:
                logger.error(f"Invalid test date format: {test_date}")
                return jsonify({'error': 'Invalid date format. Use YYYY-MM-DD'}), 400
                
        # Create new lab test record
        lab_test = LabTest(
            user_id=user_id,
            test_name=test_name,
            result=result,
            unit=unit,
            test_date=parsed_test_date,
            notes=notes
        )
        
        db.session.add(lab_test)
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'lab_test_id': lab_test.id,
            'message': 'Lab test added successfully'
        })
        
    except Exception as e:
        db.session.rollback()
        logger.error(f"Error adding lab test: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': f'An error occurred: {str(e)}'}), 500 