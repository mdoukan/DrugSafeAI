import json
import requests
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from drug_data_api import DrugDataAPI
from intelligent_drug_recommendation import IntelligentDrugRecommendation

class IntegratedDrugSystem:
    """
    Integrated system combining API data collection with intelligent recommendations
    """
    
    def __init__(self):
        self.api = DrugDataAPI()
        self.recommender = IntelligentDrugRecommendation()
        self.cache = {}
        self.cache_duration = 3600  # 1 hour cache
        
    def get_real_time_recommendations(self, 
                                    current_drug: str, 
                                    patient_data: Dict,
                                    use_api_data: bool = True) -> Dict:
        """
        Get recommendations using both cached data and real-time API data
        """
        result = {
            'timestamp': datetime.now().isoformat(),
            'current_drug': current_drug,
            'patient_data': patient_data,
            'recommendations': [],
            'api_data_used': use_api_data,
            'evidence_sources': []
        }
        
        # Get base recommendations from intelligent system
        base_recommendations = self.recommender.recommend_alternatives(current_drug, patient_data)
        
        if use_api_data:
            # Enhance with real-time API data
            enhanced_recommendations = self._enhance_with_api_data(base_recommendations, current_drug)
            result['recommendations'] = enhanced_recommendations
        else:
            result['recommendations'] = base_recommendations
        
        return result
    
    def _enhance_with_api_data(self, base_recommendations: List[Dict], current_drug: str) -> List[Dict]:
        """
        Enhance base recommendations with real-time API data
        """
        enhanced_recommendations = []
        
        for rec in base_recommendations:
            drug_name = rec['drug_name']
            enhanced_rec = rec.copy()
            
            # Get FDA data
            fda_data = self._get_cached_fda_data(drug_name)
            if fda_data:
                enhanced_rec['fda_warnings'] = self._extract_fda_warnings(fda_data)
                enhanced_rec['fda_contraindications'] = self._extract_fda_contraindications(fda_data)
                enhanced_rec['fda_dosage'] = self._extract_fda_dosage(fda_data)
            
            # Get PubMed evidence
            pubmed_evidence = self._get_cached_pubmed_evidence(drug_name)
            if pubmed_evidence:
                enhanced_rec['recent_evidence'] = pubmed_evidence
                enhanced_rec['evidence_level'] = 'Recent Literature'
            
            # Get drug interactions
            interactions = self._get_cached_interactions(drug_name)
            if interactions:
                enhanced_rec['api_interactions'] = interactions
            
            enhanced_recommendations.append(enhanced_rec)
        
        return enhanced_recommendations
    
    def _get_cached_fda_data(self, drug_name: str) -> Optional[Dict]:
        """
        Get FDA data with caching
        """
        cache_key = f"fda_{drug_name}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now().timestamp() - cache_time) < self.cache_duration:
                return data
        
        # Fetch new data
        fda_data = self.api.get_fda_drug_info(drug_name)
        if fda_data:
            self.cache[cache_key] = (datetime.now().timestamp(), fda_data)
        
        return fda_data
    
    def _get_cached_pubmed_evidence(self, drug_name: str) -> List[Dict]:
        """
        Get PubMed evidence with caching
        """
        cache_key = f"pubmed_{drug_name}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now().timestamp() - cache_time) < self.cache_duration:
                return data
        
        # Fetch new data
        query = f'"{drug_name}" AND "safety" AND "efficacy"'
        pubmed_data = self.api.search_pubmed_articles(query, max_results=3)
        if pubmed_data:
            self.cache[cache_key] = (datetime.now().timestamp(), pubmed_data)
        
        return pubmed_data
    
    def _get_cached_interactions(self, drug_name: str) -> List[Dict]:
        """
        Get drug interactions with caching
        """
        cache_key = f"interactions_{drug_name}"
        if cache_key in self.cache:
            cache_time, data = self.cache[cache_key]
            if (datetime.now().timestamp() - cache_time) < self.cache_duration:
                return data
        
        # Fetch new data
        interactions = self.api.get_drug_interactions(drug_name)
        if interactions:
            self.cache[cache_key] = (datetime.now().timestamp(), interactions)
        
        return interactions
    
    def _extract_fda_warnings(self, fda_data: Dict) -> List[str]:
        """
        Extract warnings from FDA data
        """
        warnings = []
        if 'warnings' in fda_data:
            warnings.extend(fda_data['warnings'])
        if 'boxed_warnings' in fda_data:
            warnings.extend(fda_data['boxed_warnings'])
        return warnings[:3]  # Limit to top 3 warnings
    
    def _extract_fda_contraindications(self, fda_data: Dict) -> List[str]:
        """
        Extract contraindications from FDA data
        """
        contraindications = []
        if 'contraindications' in fda_data:
            contraindications.extend(fda_data['contraindications'])
        return contraindications[:5]  # Limit to top 5
    
    def _extract_fda_dosage(self, fda_data: Dict) -> Dict:
        """
        Extract dosage information from FDA data
        """
        dosage_info = {}
        if 'dosage_and_administration' in fda_data:
            dosage_info['recommended'] = fda_data['dosage_and_administration']
        if 'dosage_forms' in fda_data:
            dosage_info['forms'] = fda_data['dosage_forms']
        return dosage_info
    
    def validate_recommendation_against_api(self, recommendation: Dict) -> Dict:
        """
        Validate a recommendation against API data
        """
        validation_result = {
            'drug_name': recommendation['drug_name'],
            'validation_score': 0.0,
            'warnings': [],
            'confirmations': [],
            'discrepancies': []
        }
        
        drug_name = recommendation['drug_name']
        
        # Check FDA data
        fda_data = self._get_cached_fda_data(drug_name)
        if fda_data:
            # Validate contraindications
            patient_conditions = recommendation.get('patient_conditions', [])
            fda_contraindications = self._extract_fda_contraindications(fda_data)
            
            for condition in patient_conditions:
                if condition in fda_contraindications:
                    validation_result['warnings'].append(f"FDA contraindication: {condition}")
                    validation_result['validation_score'] -= 0.3
                else:
                    validation_result['confirmations'].append(f"FDA: {condition} not contraindicated")
                    validation_result['validation_score'] += 0.1
            
            # Check for boxed warnings
            fda_warnings = self._extract_fda_warnings(fda_data)
            if fda_warnings:
                validation_result['warnings'].extend(fda_warnings)
                validation_result['validation_score'] -= 0.2
        
        # Check PubMed evidence
        pubmed_evidence = self._get_cached_pubmed_evidence(drug_name)
        if pubmed_evidence:
            validation_result['confirmations'].append(f"Recent evidence available ({len(pubmed_evidence)} studies)")
            validation_result['validation_score'] += 0.2
        
        # Normalize validation score
        validation_result['validation_score'] = max(0.0, min(1.0, validation_result['validation_score']))
        
        return validation_result
    
    def get_comprehensive_report(self, current_drug: str, patient_data: Dict) -> Dict:
        """
        Generate comprehensive drug recommendation report
        """
        # Get real-time recommendations
        recommendations = self.get_real_time_recommendations(current_drug, patient_data, use_api_data=True)
        
        # Validate each recommendation
        validated_recommendations = []
        for rec in recommendations['recommendations']:
            validation = self.validate_recommendation_against_api(rec)
            rec['validation'] = validation
            validated_recommendations.append(rec)
        
        # Sort by validation score
        validated_recommendations.sort(key=lambda x: x['validation']['validation_score'], reverse=True)
        
        # Generate summary
        summary = {
            'total_recommendations': len(validated_recommendations),
            'high_confidence': len([r for r in validated_recommendations if r['validation']['validation_score'] > 0.7]),
            'medium_confidence': len([r for r in validated_recommendations if 0.4 <= r['validation']['validation_score'] <= 0.7]),
            'low_confidence': len([r for r in validated_recommendations if r['validation']['validation_score'] < 0.4]),
            'warnings_count': sum(len(r['validation']['warnings']) for r in validated_recommendations)
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_drug': current_drug,
            'patient_data': patient_data,
            'summary': summary,
            'recommendations': validated_recommendations,
            'data_sources': ['FDA API', 'PubMed', 'Intelligent Algorithm'],
            'cache_status': f"{len(self.cache)} items cached"
        }

# Usage example
if __name__ == "__main__":
    # Initialize integrated system
    integrated_system = IntegratedDrugSystem()
    
    # Example patient data
    patient_data = {
        'age': 70,
        'medical_history': ['hypertension', 'diabetes', 'kidney_disease'],
        'laboratory_tests': {
            'creatinine': 1.8,
            'alt': 35
        },
        'medications': ['metformin', 'lisinopril']
    }
    
    # Get comprehensive recommendations
    report = integrated_system.get_comprehensive_report('aspirin', patient_data)
    
    print("=== INTEGRATED DRUG RECOMMENDATION SYSTEM ===")
    print(f"Patient: {patient_data['age']} years old")
    print(f"Conditions: {', '.join(patient_data['medical_history'])}")
    print(f"Current medication: {report['current_drug']}")
    
    print(f"\nSummary:")
    print(f"- Total recommendations: {report['summary']['total_recommendations']}")
    print(f"- High confidence: {report['summary']['high_confidence']}")
    print(f"- Medium confidence: {report['summary']['medium_confidence']}")
    print(f"- Low confidence: {report['summary']['low_confidence']}")
    print(f"- Warnings: {report['summary']['warnings_count']}")
    
    print(f"\nTop Recommendations:")
    for i, rec in enumerate(report['recommendations'][:3], 1):
        print(f"\n{i}. {rec['drug_name']} ({rec['class']})")
        print(f"   Validation Score: {rec['validation']['validation_score']:.2f}")
        print(f"   Recommendation Score: {rec['recommendation_score']:.3f}")
        print(f"   Reasoning: {rec['reasoning']}")
        
        if rec['validation']['warnings']:
            print(f"   ⚠️  Warnings: {', '.join(rec['validation']['warnings'][:2])}")
        
        if rec['validation']['confirmations']:
            print(f"   ✅ Confirmations: {', '.join(rec['validation']['confirmations'][:2])}")
    
    # Save comprehensive report
    with open('integrated_drug_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nComprehensive report saved to integrated_drug_report.json")
    print(f"Data sources: {', '.join(report['data_sources'])}")
    print(f"Cache status: {report['cache_status']}") 