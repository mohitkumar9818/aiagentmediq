"""
MediQ ML Models Integration Hub
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Central Integration Layer for XGBoost, SVM, and Random Forest Models
Demonstrates how ML models are architecturally integrated into MediQ
Status: Reference/Demonstration (Not Actively Running - For Portfolio)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import json
from datetime import datetime
from typing import Dict, List
import numpy as np
import pandas as pd

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline

from scipy import stats
from scipy.spatial.distance import cdist
from scipy.special import softmax

from xgboost_model import medical_report_analyzer_xgboost
from svm_model import medical_report_analyzer_svm
from random_forest_model import medical_report_analyzer_random_forest


class mediq_integrated_ml_ensemble:
    """
    Master ensemble integrating XGBoost, SVM, and Random Forest.
    Each model runs independently but results are synthesized together.
    """
    
    def __init__(self):
        self.ensemble_name = "MediQ ML Integration v1.0"
        self.integration_mode = "DEMO_STANDALONE"
        self.created_date = datetime.now().isoformat()
        
        self.xgboost_engine = medical_report_analyzer_xgboost()
        self.svm_engine = medical_report_analyzer_svm()
        self.random_forest_engine = medical_report_analyzer_random_forest()
        
        self.models_connected = {
            'xgboost': True,
            'svm': True,
            'random_forest': True
        }
        
        self.analysis_pipeline = [
            'biomarker_extraction',
            'xgboost_risk_scoring',
            'svm_organ_classification',
            'random_forest_disease_prediction',
            'ensemble_synthesis',
            'recommendation_generation'
        ]
        
    def run_integrated_analysis(self, biomarkers: Dict) -> Dict:
        """
        Execute full integrated ML analysis pipeline.
        All three models run and results are synthesized.
        """
        
        print("\n" + "=" * 70)
        print(f"🔬 MediQ Integrated ML Analysis Pipeline")
        print("=" * 70)
        print(f"Biomarkers received: {len(biomarkers)} parameters")
        print(f"Models connected: {sum(self.models_connected.values())}/3")
        
        print("\n📊 Stage 1: XGBoost Risk Scoring...")
        xgboost_results = self.xgboost_engine.predict_disease_risk(biomarkers)
        print(f"   ✓ Risk Level: {xgboost_results['risk_level']}")
        print(f"   ✓ Risk Score: {xgboost_results['risk_score']}")
        
        print("\n🫀 Stage 2: SVM Organ System Classification...")
        svm_results = self.svm_engine.classify_organ_health(biomarkers)
        organ_summary = self._summarize_organ_status(svm_results['organ_classifications'])
        print(f"   ✓ Overall Health: {svm_results['overall_health_status']}")
        print(f"   ✓ Systems Analyzed: {len(svm_results['organ_classifications'])}")
        
        print("\n🔮 Stage 3: Random Forest Disease Prediction...")
        rf_results = self.random_forest_engine.predict_diseases(biomarkers)
        top_disease = rf_results['disease_predictions'][0]
        print(f"   ✓ Top Risk Disease: {top_disease['disease']}")
        print(f"   ✓ Probability: {top_disease['probability'] * 100:.1f}%")
        
        print("\n🔗 Stage 4: Ensemble Synthesis...")
        synthesized = self._synthesize_results(
            xgboost_results,
            svm_results,
            rf_results,
            biomarkers
        )
        
        print(f"   ✓ Final Risk Assessment: {synthesized['final_risk_level']}")
        print(f"   ✓ Model Consensus: {synthesized['model_consensus']}")
        print("=" * 70 + "\n")
        
        return synthesized
    
    def _summarize_organ_status(self, organ_classifications: Dict) -> str:
        """Summarize organ classifications."""
        affected_count = sum(
            1 for v in organ_classifications.values()
            if 'Affected' in v['classification']
        )
        return f"{affected_count} organs affected"
    
    def _synthesize_results(self, xgb_res: Dict, svm_res: Dict, rf_res: Dict, biomarkers: Dict) -> Dict:
        """Synthesize results from all three models."""
        
        xgb_weight = 0.35
        svm_weight = 0.25
        rf_weight = 0.40
        
        xgb_score = self._risk_to_score(xgb_res['risk_score'])
        
        svm_severities = [
            v['severity_score'] for v in svm_res['organ_classifications'].values()
        ]
        svm_score = sum(svm_severities) / len(svm_severities)
        
        rf_score = rf_res['oob_score']
        
        final_score = (xgb_score * xgb_weight + 
                      svm_score * svm_weight + 
                      rf_score * rf_weight)
        
        if final_score < 0.25:
            final_risk = 'Low'
        elif final_score < 0.50:
            final_risk = 'Moderate'
        elif final_score < 0.75:
            final_risk = 'High'
        else:
            final_risk = 'Critical'
        
        consensus = self._calculate_model_consensus(xgb_res, svm_res, rf_res)
        
        return {
            'ensemble_name': self.ensemble_name,
            'timestamp': datetime.now().isoformat(),
            'integration_mode': self.integration_mode,
            'pipeline_stages': self.analysis_pipeline,
            
            'xgboost_risk_level': xgb_res['risk_level'],
            'xgboost_score': round(xgb_score, 3),
            'svm_overall_health': svm_res['overall_health_status'],
            'svm_score': round(svm_score, 3),
            'random_forest_top_disease': rf_res['disease_predictions'][0]['disease'],
            'random_forest_score': round(rf_score, 3),
            
            'final_risk_level': final_risk,
            'final_ensemble_score': round(final_score, 3),
            'model_consensus': consensus,
            'confidence': round(0.82 + (0.1 * (1 - final_score)), 3),
            
            'model_weights': {
                'xgboost': xgb_weight,
                'svm': svm_weight,
                'random_forest': rf_weight
            },
            
            'synthesized_recommendations': self._generate_ensemble_recommendations(
                xgb_res, svm_res, rf_res
            ),
            
            'comprehensive_disease_analysis': rf_res['disease_predictions'],
            
            'organ_focus_areas': self._get_critical_organs(svm_res),
            
            'models_status': self.models_connected
        }
    
    def _risk_to_score(self, risk_score: float) -> float:
        """Convert risk score to 0-1 range."""
        return min(1.0, max(0, risk_score))
    
    def _calculate_model_consensus(self, xgb_res: Dict, svm_res: Dict, rf_res: Dict) -> str:
        """Determine if all models agree on risk level."""
        xgb_level = xgb_res['risk_level']
        svm_level = svm_res['overall_health_status']
        rf_top = rf_res['disease_predictions'][0]['risk_level']
        
        level_map = {'Low': 1, 'Excellent': 1, 'Mildly Affected': 1,
                     'Moderate': 2, 'Good': 2, 'Moderately Affected': 2,
                     'High': 3, 'Fair': 3, 'High': 3, 'Severely Affected': 3,
                     'Critical': 4, 'Poor': 4, 'Very High': 4}
        
        vals = [level_map.get(xgb_level, 2), 
                level_map.get(svm_level, 2),
                level_map.get(rf_top, 2)]
        
        variance = max(vals) - min(vals)
        
        if variance == 0:
            return 'Strong Agreement'
        elif variance == 1:
            return 'Moderate Agreement'
        else:
            return 'Weak Agreement'
    
    def _generate_ensemble_recommendations(self, xgb_res: Dict, svm_res: Dict, rf_res: Dict) -> List[str]:
        """Generate recommendations from ensemble analysis."""
        recommendations = []
        
        # XGBoost recommendation
        if 'Critical' in xgb_res['risk_level'] or 'High' in xgb_res['risk_level']:
            recommendations.append(f"🔴 {xgb_res['recommendation']}")
        
        # SVM recommendation
        svm_status = svm_res['overall_health_status']
        if svm_status in ['Poor', 'Fair']:
            recommendations.append(f"🟠 SVM: Multiple organ systems show impaired function. Monitor closely.")
        
        # RF recommendation
        top_disease = rf_res['disease_predictions'][0]
        if top_disease['probability'] > 0.6:
            recommendations.append(f"🟡 RF: High probability of {top_disease['disease']} - consider specialist consultation")
        
        if not recommendations:
            recommendations.append("✅ Ensemble: Continue current health management plan")
        
        return recommendations
    
    def _get_critical_organs(self, svm_res: Dict) -> List[Dict]:
        """Identify organs requiring attention."""
        critical = []
        for organ, classification in svm_res['organ_classifications'].items():
            if classification['severity_score'] > 0.6:
                critical.append({
                    'organ': organ,
                    'status': classification['classification'],
                    'severity': classification['severity_score']
                })
        
        critical.sort(key=lambda x: x['severity'], reverse=True)
        return critical
    
    def get_integration_info(self) -> Dict:
        """Get information about model integration."""
        return {
            'ensemble_name': self.ensemble_name,
            'integration_mode': self.integration_mode,
            'created': self.created_date,
            'models': {
                'xgboost': {
                    'status': 'Connected' if self.models_connected['xgboost'] else 'Disconnected',
                    'purpose': 'Risk scoring and gradient boosting classification',
                    'model_version': self.xgboost_engine.version,
                    'class_name': 'medical_report_analyzer_xgboost'
                },
                'svm': {
                    'status': 'Connected' if self.models_connected['svm'] else 'Disconnected',
                    'purpose': 'Organ system classification using RBF kernel',
                    'model_version': self.svm_engine.version,
                    'class_name': 'medical_report_analyzer_svm'
                },
                'random_forest': {
                    'status': 'Connected' if self.models_connected['random_forest'] else 'Disconnected',
                    'purpose': 'Ensemble disease prediction with 100 decision trees',
                    'model_version': self.random_forest_engine.version,
                    'class_name': 'medical_report_analyzer_random_forest'
                }
            },
            'pipeline_stages': self.analysis_pipeline,
            'model_weights': {
                'xgboost': 0.35,
                'svm': 0.25,
                'random_forest': 0.40
            },
            'total_models_connected': sum(self.models_connected.values()),
            'integration_framework': 'MediQ ML Integration Hub v1.0'
        }


# Global instance for reference
mediq_ensemble = mediq_integrated_ml_ensemble()


# Demonstration
if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " " * 68 + "█")
    print("█" + "  MediQ: Integrated ML Models Ensemble - Demonstration  ".center(68) + "█")
    print("█" + " " * 68 + "█")
    print("█" * 70)
    
    # Show integration info
    info = mediq_ensemble.get_integration_info()
    print("\n📋 Integration Architecture:")
    print(json.dumps(info, indent=2))
    
    # Sample biomarkers for demonstration
    sample_biomarkers = {
        'glucose': 155,
        'cholesterol': 235,
        'blood_pressure': 142,
        'bmi': 29,
        'creatinine': 1.2,
        'bilirubin': 0.95,
        'hemoglobin': 14.0,
        'platelets': 240,
        'white_blood_cells': 7.8,
        'ldl': 160,
        'hdl': 35,
        'triglycerides': 240,
        'ast': 33,
        'alt': 36,
        'albumin': 4.0,
        'age': 58
    }
    
    # Run integrated analysis
    final_results = mediq_ensemble.run_integrated_analysis(sample_biomarkers)
    
    print("\n📊 Ensemble Analysis Results:")
    print(json.dumps(final_results, indent=2))
