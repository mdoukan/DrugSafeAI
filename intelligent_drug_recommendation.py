import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import json
import requests
from datetime import datetime

class IntelligentDrugRecommendation:
    """
    Intelligent drug recommendation system using ML and real-time data
    """
    
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.drug_database = {}
        self.patient_profiles = {}
        self.interaction_matrix = None
        
    def build_drug_database(self):
        """
        Build comprehensive drug database from multiple sources
        """
        # Core drug information
        self.drug_database = {
            'aspirin': {
                'class': 'NSAID',
                'mechanism': 'COX inhibitor',
                'indications': ['pain', 'fever', 'cardiovascular_protection'],
                'contraindications': ['bleeding_disorders', 'peptic_ulcer'],
                'side_effects': ['gastrointestinal_bleeding', 'allergic_reactions'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'half_life': 15,  # minutes
                'protein_binding': 0.99,
                'bioavailability': 0.68
            },
            'naproxen': {
                'class': 'NSAID',
                'mechanism': 'COX inhibitor',
                'indications': ['pain', 'inflammation', 'arthritis'],
                'contraindications': ['bleeding_disorders', 'heart_failure'],
                'side_effects': ['gastrointestinal_bleeding', 'cardiovascular_risk'],
                'metabolism': 'hepatic',
                'excretion': 'renal',
                'half_life': 14,  # hours
                'protein_binding': 0.99,
                'bioavailability': 0.95
            },
            'metformin': {
                'class': 'Biguanide',
                'mechanism': 'AMPK activator',
                'indications': ['diabetes_type_2'],
                'contraindications': ['kidney_disease', 'lactic_acidosis'],
                'side_effects': ['gastrointestinal_distress', 'vitamin_b12_deficiency'],
                'metabolism': 'minimal',
                'excretion': 'renal',
                'half_life': 6.2,  # hours
                'protein_binding': 0.0,
                'bioavailability': 0.55
            },
            'lisinopril': {
                'class': 'ACE_inhibitor',
                'mechanism': 'ACE inhibitor',
                'indications': ['hypertension', 'heart_failure'],
                'contraindications': ['pregnancy', 'angioedema'],
                'side_effects': ['cough', 'hyperkalemia', 'renal_impairment'],
                'metabolism': 'minimal',
                'excretion': 'renal',
                'half_life': 12,  # hours
                'protein_binding': 0.0,
                'bioavailability': 0.25
            }
        }
        
        # Build interaction matrix
        self._build_interaction_matrix()
    
    def _build_interaction_matrix(self):
        """
        Build drug-drug interaction matrix
        """
        drugs = list(self.drug_database.keys())
        n_drugs = len(drugs)
        
        # Initialize interaction matrix
        self.interaction_matrix = np.zeros((n_drugs, n_drugs))
        
        # Define interactions (0 = no interaction, 1 = mild, 2 = moderate, 3 = severe)
        interactions = {
            ('aspirin', 'naproxen'): 2,  # Both NSAIDs, increased bleeding risk
            ('aspirin', 'metformin'): 1,  # Mild interaction
            ('aspirin', 'lisinopril'): 1,  # Mild interaction
            ('naproxen', 'metformin'): 1,  # Mild interaction
            ('naproxen', 'lisinopril'): 1,  # Mild interaction
            ('metformin', 'lisinopril'): 2,  # Moderate interaction, renal effects
        }
        
        # Fill interaction matrix
        for (drug1, drug2), severity in interactions.items():
            if drug1 in drugs and drug2 in drugs:
                i, j = drugs.index(drug1), drugs.index(drug2)
                self.interaction_matrix[i][j] = severity
                self.interaction_matrix[j][i] = severity  # Symmetric
    
    def calculate_drug_similarity(self, drug1: str, drug2: str) -> float:
        """
        Calculate similarity between two drugs based on multiple factors
        """
        if drug1 not in self.drug_database or drug2 not in self.drug_database:
            return 0.0
        
        d1, d2 = self.drug_database[drug1], self.drug_database[drug2]
        
        # Class similarity
        class_similarity = 1.0 if d1['class'] == d2['class'] else 0.0
        
        # Mechanism similarity
        mechanism_similarity = 1.0 if d1['mechanism'] == d2['mechanism'] else 0.0
        
        # Indication overlap
        indications1 = set(d1['indications'])
        indications2 = set(d2['indications'])
        indication_similarity = len(indications1 & indications2) / len(indications1 | indications2) if indications1 | indications2 else 0.0
        
        # Side effect similarity
        side_effects1 = set(d1['side_effects'])
        side_effects2 = set(d2['side_effects'])
        side_effect_similarity = len(side_effects1 & side_effects2) / len(side_effects1 | side_effects2) if side_effects1 | side_effects2 else 0.0
        
        # Pharmacokinetic similarity
        pk_similarity = 1.0 - abs(d1['half_life'] - d2['half_life']) / max(d1['half_life'], d2['half_life'])
        
        # Weighted average
        weights = [0.3, 0.2, 0.2, 0.2, 0.1]  # Class, mechanism, indications, side effects, PK
        similarities = [class_similarity, mechanism_similarity, indication_similarity, side_effect_similarity, pk_similarity]
        
        return np.average(similarities, weights=weights)
    
    def calculate_patient_risk_score(self, patient_data: Dict, drug: str) -> float:
        """
        Calculate patient-specific risk score for a drug
        """
        if drug not in self.drug_database:
            return 1.0  # High risk if drug not in database
        
        drug_info = self.drug_database[drug]
        risk_score = 0.0
        
        # Age-related risks
        age = patient_data.get('age', 50)
        if age > 65:
            risk_score += 0.2  # Elderly patients have higher risk
        
        # Medical history risks
        medical_history = patient_data.get('medical_history', [])
        contraindications = set(drug_info['contraindications'])
        
        for condition in medical_history:
            if condition in contraindications:
                risk_score += 0.5  # Major contraindication
            elif condition in drug_info['side_effects']:
                risk_score += 0.3  # Increased risk of side effect
        
        # Lab value risks
        lab_tests = patient_data.get('laboratory_tests', {})
        
        # Kidney function
        if 'creatinine' in lab_tests:
            creatinine = lab_tests['creatinine']
            if creatinine > 1.5 and drug_info['excretion'] == 'renal':
                risk_score += 0.3
        
        # Liver function
        if 'alt' in lab_tests:
            alt = lab_tests['alt']
            if alt > 40 and drug_info['metabolism'] == 'hepatic':
                risk_score += 0.2
        
        # Current medications
        current_meds = patient_data.get('medications', [])
        for med in current_meds:
            if med in self.drug_database:
                interaction_severity = self.get_interaction_severity(drug, med)
                risk_score += interaction_severity * 0.2
        
        return min(risk_score, 1.0)  # Cap at 1.0
    
    def get_interaction_severity(self, drug1: str, drug2: str) -> float:
        """
        Get interaction severity between two drugs
        """
        if drug1 not in self.drug_database or drug2 not in self.drug_database:
            return 0.0
        
        drugs = list(self.drug_database.keys())
        i, j = drugs.index(drug1), drugs.index(drug2)
        
        return self.interaction_matrix[i][j] / 3.0  # Normalize to 0-1
    
    def recommend_alternatives(self, 
                             current_drug: str, 
                             patient_data: Dict, 
                             max_alternatives: int = 5) -> List[Dict]:
        """
        Recommend alternative drugs based on patient characteristics
        """
        if current_drug not in self.drug_database:
            return []
        
        current_drug_info = self.drug_database[current_drug]
        alternatives = []
        
        for drug, drug_info in self.drug_database.items():
            if drug == current_drug:
                continue
            
            # Calculate similarity to current drug
            similarity = self.calculate_drug_similarity(current_drug, drug)
            
            # Calculate patient-specific risk
            risk_score = self.calculate_patient_risk_score(patient_data, drug)
            
            # Calculate overall recommendation score
            # Higher similarity and lower risk = better recommendation
            recommendation_score = similarity * (1.0 - risk_score)
            
            # Check if drug is suitable for patient's condition
            patient_conditions = patient_data.get('medical_history', [])
            contraindicated = any(condition in drug_info['contraindications'] for condition in patient_conditions)
            
            if not contraindicated and recommendation_score > 0.1:
                alternatives.append({
                    'drug_name': drug,
                    'class': drug_info['class'],
                    'similarity_score': similarity,
                    'risk_score': risk_score,
                    'recommendation_score': recommendation_score,
                    'indications': drug_info['indications'],
                    'side_effects': drug_info['side_effects'],
                    'mechanism': drug_info['mechanism'],
                    'reasoning': self._generate_recommendation_reasoning(
                        current_drug, drug, similarity, risk_score, patient_data
                    )
                })
        
        # Sort by recommendation score and return top alternatives
        alternatives.sort(key=lambda x: x['recommendation_score'], reverse=True)
        return alternatives[:max_alternatives]
    
    def _generate_recommendation_reasoning(self, 
                                         current_drug: str, 
                                         alternative: str, 
                                         similarity: float, 
                                         risk_score: float, 
                                         patient_data: Dict) -> str:
        """
        Generate human-readable reasoning for recommendation
        """
        current_info = self.drug_database[current_drug]
        alt_info = self.drug_database[alternative]
        
        reasoning_parts = []
        
        # Similarity reasoning
        if similarity > 0.7:
            reasoning_parts.append(f"Similar mechanism of action ({alt_info['mechanism']})")
        elif similarity > 0.4:
            reasoning_parts.append(f"Related drug class ({alt_info['class']})")
        
        # Risk reasoning
        if risk_score < 0.3:
            reasoning_parts.append("Lower risk profile for this patient")
        elif risk_score < 0.6:
            reasoning_parts.append("Moderate risk, requires monitoring")
        
        # Patient-specific reasoning
        age = patient_data.get('age', 50)
        if age > 65 and alt_info['half_life'] < current_info['half_life']:
            reasoning_parts.append("Shorter half-life may be safer in elderly patients")
        
        # Interaction reasoning
        current_meds = patient_data.get('medications', [])
        for med in current_meds:
            if med in self.drug_database:
                interaction = self.get_interaction_severity(alternative, med)
                if interaction < 0.3:
                    reasoning_parts.append(f"Fewer interactions with {med}")
        
        return "; ".join(reasoning_parts) if reasoning_parts else "Alternative treatment option"
    
    def train_recommendation_model(self, training_data: List[Dict]):
        """
        Train ML model on historical patient outcomes
        """
        # This would require historical data of patient outcomes
        # For now, we'll use a simple rule-based approach
        pass
    
    def update_recommendations_from_feedback(self, feedback_data: List[Dict]):
        """
        Update recommendation system based on user feedback
        """
        for feedback in feedback_data:
            drug = feedback.get('drug_name')
            feedback_type = feedback.get('feedback_type')
            severity = feedback.get('severity')
            
            if drug in self.drug_database:
                # Adjust risk factors based on feedback
                if feedback_type == 'incorrect_risk_factor' and severity == 'critical':
                    # Increase risk score for this drug
                    pass
                elif feedback_type == 'missing_alternative':
                    # Add new alternative to database
                    pass

# Usage example
if __name__ == "__main__":
    # Initialize intelligent recommendation system
    recommender = IntelligentDrugRecommendation()
    recommender.build_drug_database()
    
    # Example patient data
    patient_data = {
        'age': 70,
        'medical_history': ['hypertension', 'diabetes'],
        'laboratory_tests': {
            'creatinine': 1.8,
            'alt': 35
        },
        'medications': ['metformin', 'lisinopril']
    }
    
    # Get recommendations for aspirin
    recommendations = recommender.recommend_alternatives('aspirin', patient_data)
    
    print("=== INTELLIGENT DRUG RECOMMENDATIONS ===")
    print(f"Patient: {patient_data['age']} years old, {', '.join(patient_data['medical_history'])}")
    print(f"Current medication: aspirin")
    print("\nRecommended alternatives:")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['drug_name']} ({rec['class']})")
        print(f"   Recommendation Score: {rec['recommendation_score']:.3f}")
        print(f"   Similarity: {rec['similarity_score']:.3f}")
        print(f"   Risk Score: {rec['risk_score']:.3f}")
        print(f"   Reasoning: {rec['reasoning']}")
    
    # Save recommendations
    with open('intelligent_recommendations.json', 'w') as f:
        json.dump({
            'patient_data': patient_data,
            'current_drug': 'aspirin',
            'recommendations': recommendations
        }, f, indent=2)
    
    print(f"\nRecommendations saved to intelligent_recommendations.json") 