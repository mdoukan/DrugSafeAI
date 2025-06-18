import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple
import logging

class DrugDataValidator:
    """
    System to validate and cross-reference drug data from multiple sources
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_results = []
        
    def validate_risk_factors(self, drug_data: Dict) -> List[Dict]:
        """
        Validate risk factors for consistency and reasonableness
        """
        issues = []
        
        for drug_name, alternatives in drug_data.items():
            if drug_name == 'default':
                continue
                
            # Check for consistency in risk factors
            risk_factors = []
            for category in ['same_class', 'different_class']:
                if category in alternatives:
                    for alt in alternatives[category]:
                        if 'risk_factor' in alt:
                            risk_factors.append({
                                'alternative': alt['name'],
                                'risk_factor': alt['risk_factor'],
                                'category': category
                            })
            
            # Validate risk factor ranges
            for rf in risk_factors:
                if not (0 <= rf['risk_factor'] <= 1):
                    issues.append({
                        'type': 'INVALID_RISK_FACTOR',
                        'drug': drug_name,
                        'alternative': rf['alternative'],
                        'risk_factor': rf['risk_factor'],
                        'message': f"Risk factor {rf['risk_factor']} is not between 0 and 1"
                    })
            
            # Check for logical consistency
            if len(risk_factors) > 1:
                # Same class alternatives should have similar risk factors
                same_class_rfs = [rf for rf in risk_factors if rf['category'] == 'same_class']
                if len(same_class_rfs) > 1:
                    rf_values = [rf['risk_factor'] for rf in same_class_rfs]
                    if max(rf_values) - min(rf_values) > 0.3:  # More than 30% difference
                        issues.append({
                            'type': 'INCONSISTENT_SAME_CLASS',
                            'drug': drug_name,
                            'risk_factors': same_class_rfs,
                            'message': f"Same class alternatives have very different risk factors: {rf_values}"
                        })
    
        return issues
    
    def validate_evidence_quality(self, drug_data: Dict) -> List[Dict]:
        """
        Validate the quality and recency of evidence
        """
        issues = []
        current_year = datetime.now().year
        
        for drug_name, alternatives in drug_data.items():
            if drug_name == 'default':
                continue
                
            for category in ['same_class', 'different_class']:
                if category in alternatives:
                    for alt in alternatives[category]:
                        if 'evidence' in alt:
                            evidence = alt['evidence']
                            
                            # Check if evidence is too old
                            if 'JAMA' in evidence or 'NEJM' in evidence or 'Lancet' in evidence:
                                # Extract year from evidence string
                                year = self._extract_year_from_evidence(evidence)
                                if year and current_year - year > 10:
                                    issues.append({
                                        'type': 'OUTDATED_EVIDENCE',
                                        'drug': drug_name,
                                        'alternative': alt['name'],
                                        'evidence': evidence,
                                        'year': year,
                                        'age': current_year - year,
                                        'message': f"Evidence is {current_year - year} years old"
                                    })
                            
                            # Check for missing evidence
                            if not evidence or evidence == 'General recommendation':
                                issues.append({
                                    'type': 'MISSING_EVIDENCE',
                                    'drug': drug_name,
                                    'alternative': alt['name'],
                                    'message': "No specific evidence provided"
                                })
        
        return issues
    
    def _extract_year_from_evidence(self, evidence: str) -> int:
        """
        Extract year from evidence string like 'JAMA 2020;323(2):156-158'
        """
        import re
        year_match = re.search(r'20\d{2}', evidence)
        if year_match:
            return int(year_match.group())
        return None
    
    def cross_reference_data(self, current_data: Dict, external_data: Dict) -> List[Dict]:
        """
        Cross-reference current data with external sources
        """
        discrepancies = []
        
        for drug_name, current_alternatives in current_data.items():
            if drug_name == 'default':
                continue
                
            if drug_name in external_data:
                external_alternatives = external_data[drug_name]
                
                # Compare risk factors
                for category in ['same_class', 'different_class']:
                    if category in current_alternatives and category in external_alternatives:
                        current_alts = {alt['name']: alt for alt in current_alternatives[category]}
                        external_alts = {alt['name']: alt for alt in external_alternatives[category]}
                        
                        # Find common alternatives
                        common_alternatives = set(current_alts.keys()) & set(external_alts.keys())
                        
                        for alt_name in common_alternatives:
                            current_rf = current_alts[alt_name].get('risk_factor')
                            external_rf = external_alts[alt_name].get('risk_factor')
                            
                            if current_rf and external_rf:
                                difference = abs(current_rf - external_rf)
                                if difference > 0.1:  # More than 10% difference
                                    discrepancies.append({
                                        'type': 'RISK_FACTOR_DISCREPANCY',
                                        'drug': drug_name,
                                        'alternative': alt_name,
                                        'current_risk_factor': current_rf,
                                        'external_risk_factor': external_rf,
                                        'difference': difference,
                                        'message': f"Risk factor differs by {difference:.2f}"
                                    })
        
        return discrepancies
    
    def generate_validation_report(self, drug_data: Dict, external_data: Dict = None) -> Dict:
        """
        Generate comprehensive validation report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {},
            'issues': [],
            'recommendations': []
        }
        
        # Validate risk factors
        risk_issues = self.validate_risk_factors(drug_data)
        report['issues'].extend(risk_issues)
        
        # Validate evidence quality
        evidence_issues = self.validate_evidence_quality(drug_data)
        report['issues'].extend(evidence_issues)
        
        # Cross-reference with external data if available
        if external_data:
            cross_ref_issues = self.cross_reference_data(drug_data, external_data)
            report['issues'].extend(cross_ref_issues)
        
        # Generate summary
        total_issues = len(report['issues'])
        issue_types = {}
        for issue in report['issues']:
            issue_type = issue['type']
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        report['summary'] = {
            'total_issues': total_issues,
            'issue_types': issue_types,
            'severity_levels': {
                'critical': len([i for i in report['issues'] if i['type'] in ['INVALID_RISK_FACTOR']]),
                'warning': len([i for i in report['issues'] if i['type'] in ['OUTDATED_EVIDENCE', 'RISK_FACTOR_DISCREPANCY']]),
                'info': len([i for i in report['issues'] if i['type'] in ['MISSING_EVIDENCE', 'INCONSISTENT_SAME_CLASS']])
            }
        }
        
        # Generate recommendations
        if total_issues > 0:
            report['recommendations'] = [
                "Review and update outdated evidence (older than 10 years)",
                "Validate risk factors against recent clinical studies",
                "Add missing evidence for alternatives without references",
                "Standardize risk factor calculations across similar drug classes",
                "Implement automated data validation in the development pipeline"
            ]
        
        return report
    
    def save_validation_report(self, report: Dict, filename: str = None):
        """
        Save validation report to file
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"drug_data_validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Validation report saved to {filename}")
        return filename

# Usage example
if __name__ == "__main__":
    # Load current drug data
    from app.services.prediction_service import PredictionService
    
    prediction_service = PredictionService()
    current_data = prediction_service._get_alternative_recommendations.__defaults__[0]  # This is a simplified approach
    
    # Initialize validator
    validator = DrugDataValidator()
    
    # Generate validation report
    report = validator.generate_validation_report(current_data)
    
    # Save report
    filename = validator.save_validation_report(report)
    
    # Print summary
    print("=== DRUG DATA VALIDATION REPORT ===")
    print(f"Total issues found: {report['summary']['total_issues']}")
    print(f"Issue types: {report['summary']['issue_types']}")
    print(f"Severity levels: {report['summary']['severity_levels']}")
    
    if report['recommendations']:
        print("\nRecommendations:")
        for rec in report['recommendations']:
            print(f"- {rec}")
    
    print(f"\nDetailed report saved to: {filename}") 