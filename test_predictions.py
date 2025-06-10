from app.services.prediction_service import PredictionService

def main():
    ps = PredictionService()
    
    # Test different medications with the same patient profile
    patient_data = {
        'age': 40,
        'weight': 70,
        'height': 175,
        'sex': 1,
        'serious': 0,
        'medications': []
    }
    
    medications = ['aspirin', 'ibuprofen', 'omeprazole', 'atorvastatin', 'metformin']
    
    print("Testing different medications with the same patient profile:")
    print("-" * 60)
    
    for med in medications:
        patient_data['medications'] = [med]
        result = ps.predict_side_effects(patient_data)
        
        if 'results' in result and len(result['results']) > 0:
            med_result = result['results'][0]
            print(f"Medication: {med_result['medication']}")
            print(f"Risk Level: {med_result['risk_level']}")
            print(f"Probability: {med_result['probability']:.4f}")
            print("-" * 60)
        else:
            print(f"No results for {med}")
            print("-" * 60)
    
    # Test serious condition
    print("\nTesting with serious condition:")
    print("-" * 60)
    
    patient_data['serious'] = 1
    patient_data['medications'] = ['aspirin']
    result = ps.predict_side_effects(patient_data)
    
    if 'results' in result and len(result['results']) > 0:
        med_result = result['results'][0]
        print(f"Medication: {med_result['medication']} (with serious condition)")
        print(f"Risk Level: {med_result['risk_level']}")
        print(f"Probability: {med_result['probability']:.4f}")
        print("-" * 60)

if __name__ == "__main__":
    main() 