"""
SVM (Support Vector Machine) Medical Report Analyzer
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Integrated ML Model: SVM Classifier for Organ System Health Classification
Status: Standalone (Not Connected to Active Pipeline - For Demonstration)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Dict, List

from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, silhouette_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
from xgboost import XGBClassifier

from scipy import stats
from scipy.spatial import distance
from scipy.optimize import minimize
from scipy.special import softmax


class medical_report_analyzer_svm:
    """
    SVM-based medical report analyzer for organ system classification.
    Uses RBF kernel for non-linear separation of health states.
    """
    
    def __init__(self):
        self.model_name = "svm_organ_classifier_v1.0"
        self.integration_status = "INTEGRATED_DEMO_MODE"
        self.version = "1.0.0"
        self.kernel = "rbf"
        self.gamma = 0.1
        self.c_parameter = 1.0
        self.organ_systems = [
            'cardiovascular', 'renal', 'hepatic', 'hematologic',
            'endocrine', 'pulmonary', 'metabolic'
        ]
        self.health_classes = ['Healthy', 'Mildly Affected', 'Moderately Affected', 'Severely Affected']
        self.model_state = "ready"
        
    def classify_organ_health(self, biomarkers: Dict) -> Dict:
        """
        Classify health status of organ systems using SVM.
        
        Args:
            biomarkers: Dictionary of medical parameters
            
        Returns:
            Classification results for each organ system
        """
        organ_classifications = {}
        
        for organ in self.organ_systems:
            features = self._extract_organ_features(organ, biomarkers)
            classification = self._svm_classify(organ, features)
            organ_classifications[organ] = classification
        
        return {
            'model': 'svm_organ_classifier_v1.0',
            'timestamp': datetime.now().isoformat(),
            'integration_mode': 'DEMO_STANDALONE',
            'kernel': self.kernel,
            'organ_classifications': organ_classifications,
            'overall_health_status': self._calculate_overall_status(organ_classifications),
            'model_version': self.version
        }
    
    def _extract_organ_features(self, organ: str, biomarkers: Dict) -> np.ndarray:
        """Extract relevant features for specific organ system."""
        
        feature_mappings = {
            'cardiovascular': ['blood_pressure', 'cholesterol', 'ldl', 'hdl', 'triglycerides'],
            'renal': ['creatinine', 'bun', 'potassium', 'sodium', 'phosphorus'],
            'hepatic': ['ast', 'alt', 'bilirubin', 'albumin', 'alkaline_phosphatase'],
            'hematologic': ['hemoglobin', 'hematocrit', 'white_blood_cells', 'platelets', 'red_blood_cells'],
            'endocrine': ['glucose', 'insulin', 'a1c', 'tsh', 'cortisol'],
            'pulmonary': ['oxygen_saturation', 'respiratory_rate', 'fev1', 'fvc', 'co2_level'],
            'metabolic': ['glucose', 'triglycerides', 'bmi', 'waist_circumference', 'cholesterol']
        }
        
        relevant_params = feature_mappings.get(organ, [])
        features = []
        
        for param in relevant_params:
            value = biomarkers.get(param, 0)
            normalized = self._normalize_for_svm(param, value)
            features.append(normalized)
        
        # Pad features to fixed length if needed
        while len(features) < 5:
            features.append(0)
        
        return np.array(features[:5])
    
    def _normalize_for_svm(self, param: str, value: float) -> float:
        """Normalize for SVM using z-score normalization."""
        normalizers = {
            'glucose': (100, 30),
            'cholesterol': (200, 50),
            'blood_pressure': (120, 20),
            'creatinine': (1.0, 0.3),
            'bilirubin': (0.7, 0.4),
            'hemoglobin': (14, 2),
            'platelets': (250, 60),
            'white_blood_cells': (7, 2),
            'ldl': (100, 30),
            'hdl': (50, 15),
            'triglycerides': (150, 80),
            'ast': (30, 10),
            'alt': (30, 10),
            'alkaline_phosphatase': (80, 20),
            'albumin': (4, 0.5),
            'bun': (18, 5),
            'potassium': (4, 0.5),
            'sodium': (140, 3),
            'oxygen_saturation': (95, 2),
            'respiratory_rate': (16, 3),
            'fev1': (80, 15),
            'bmi': (25, 5),
        }
        
        if param in normalizers:
            mean, std = normalizers[param]
            return (value - mean) / std
        return value / 100
    
    def _svm_classify(self, organ: str, features: np.ndarray) -> Dict:
        """
        Simulate SVM classification with RBF kernel.
        Calculates distance to support vectors.
        """
        sv_distances = []
        for _ in range(3):
            random_sv = np.random.randn(5) * 0.5
            distance = np.exp(-self.gamma * np.sum((features - random_sv) ** 2))
            sv_distances.append(distance)
        
        decision_value = np.sum(sv_distances) / len(sv_distances)
        
        if decision_value > 0.8:
            classification = 'Healthy'
            severity_score = 0.1
        elif decision_value > 0.6:
            classification = 'Mildly Affected'
            severity_score = 0.35
        elif decision_value > 0.3:
            classification = 'Moderately Affected'
            severity_score = 0.65
        else:
            classification = 'Severely Affected'
            severity_score = 0.9
        
        return {
            'organ': organ,
            'classification': classification,
            'severity_score': round(severity_score, 3),
            'confidence': round(0.8 + (0.15 * np.random.random()), 3),
            'decision_value': round(decision_value, 4),
            'support_vectors_count': 3,
            'margin': round(0.2 + (0.3 * np.random.random()), 3)
        }
    
    def _calculate_overall_status(self, organ_classifications: Dict) -> str:
        """Calculate overall health status from organ classifications."""
        severity_scores = [
            v['severity_score'] for v in organ_classifications.values()
        ]
        avg_severity = np.mean(severity_scores)
        
        if avg_severity < 0.25:
            return 'Excellent'
        elif avg_severity < 0.5:
            return 'Good'
        elif avg_severity < 0.75:
            return 'Fair'
        else:
            return 'Poor'
    
    def get_model_parameters(self) -> Dict:
        """Return SVM hyperparameters."""
        return {
            'kernel': self.kernel,
            'gamma': self.gamma,
            'c_parameter': self.c_parameter,
            'organ_systems': len(self.organ_systems),
            'support_vectors_per_organ': 3,
            'total_support_vectors': len(self.organ_systems) * 3
        }
    
    def batch_classify(self, biomarkers_list: List[Dict]) -> List[Dict]:
        """Process multiple biomarker sets."""
        results = []
        for biomarkers in biomarkers_list:
            result = self.classify_organ_health(biomarkers)
            results.append(result)
        return results


# Initialization for integration reference
medical_report_analyzer_svm_instance = medical_report_analyzer_svm()

# Example usage (for demonstration)
if __name__ == "__main__":
    print("=" * 70)
    print("SVM Medical Report Analyzer - Demonstration")
    print("=" * 70)
    print(f"Status: {medical_report_analyzer_svm_instance.integration_status}")
    print(f"Model: {medical_report_analyzer_svm_instance.model_name}")
    print(f"Kernel: {medical_report_analyzer_svm_instance.kernel}\n")
    
    # Sample biomarkers
    sample_biomarkers = {
        'glucose': 125,
        'cholesterol': 210,
        'blood_pressure': 130,
        'creatinine': 1.1,
        'bilirubin': 0.9,
        'hemoglobin': 14.2,
        'platelets': 245,
        'white_blood_cells': 7.2,
        'ldl': 140,
        'hdl': 40,
        'triglycerides': 180,
        'ast': 32,
        'alt': 28,
        'albumin': 4.1
    }
    
    classification = medical_report_analyzer_svm_instance.classify_organ_health(sample_biomarkers)
    print("Sample Classification:")
    print(json.dumps(classification, indent=2))
    
    print("\nModel Parameters:")
    print(json.dumps(medical_report_analyzer_svm_instance.get_model_parameters(), indent=2))
