"""
Privacy Control Framework Demo

Comprehensive demonstration of the privacy control system for fraud detection:
- Data anonymization and pseudonymization
- Consent management and tracking
- GDPR compliance management
- Access control and authorization
- Data classification and sensitivity analysis
"""

import sys
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from privacy.anonymizer import DataAnonymizer, AnonymizationConfig
from privacy.consent_manager import ConsentManager, ConsentPurpose, ConsentStatus
from privacy.gdpr_compliance import GDPRComplianceManager, GDPRRight, BreachSeverity
from privacy.access_control import PrivacyAccessController, UserRole, AccessPurpose, DataSensitivity
from privacy.data_classifier import DataClassifier, DataCategory

class PrivacyFrameworkDemo:
    """
    Comprehensive demonstration of the privacy control framework
    """
    
    def __init__(self):
        print("🔐 Initializing Privacy Control Framework...")
        
        # Initialize all privacy components
        self.anonymizer = DataAnonymizer(AnonymizationConfig(
            preserve_format=True,
            reversible=False,
            mask_char='*'
        ))
        
        self.consent_manager = ConsentManager()
        self.gdpr_manager = GDPRComplianceManager()
        self.access_controller = PrivacyAccessController()
        self.data_classifier = DataClassifier()
        
        print("✅ Privacy Framework initialized successfully!")
    
    def demo_data_anonymization(self):
        """
        Demonstrate data anonymization capabilities
        """
        print("\n" + "="*60)
        print("🎭 DATA ANONYMIZATION DEMONSTRATION")
        print("="*60)
        
        # Sample sensitive data
        sample_data = {
            'customer_name': 'John Doe',
            'email': 'john.doe@example.com',
            'phone': '+1-555-123-4567',
            'ssn': '123-45-6789',
            'credit_card': '4532-1234-5678-9012',
            'ip_address': '192.168.1.100',
            'address': '123 Main St, New York, NY 10001',
            'transaction_amount': 1250.75,
            'notes': 'Customer called about suspicious activity on account'
        }
        
        print("📋 Original Data:")
        for field, value in sample_data.items():
            print(f"  {field}: {value}")
        
        # Define field mapping for anonymization
        field_mapping = {
            'customer_name': 'name',
            'email': 'email',
            'phone': 'phone',
            'ssn': 'ssn',
            'credit_card': 'credit_card',
            'ip_address': 'ip_address',
            'address': 'address'
        }
        
        # Anonymize the data
        anonymized_data = self.anonymizer.anonymize_record(sample_data, field_mapping)
        
        print("\n🔒 Anonymized Data:")
        for field, value in anonymized_data.items():
            print(f"  {field}: {value}")
        
        # Generate anonymization summary
        summary = self.anonymizer.get_anonymization_summary(sample_data, anonymized_data)
        print(f"\n📊 Anonymization Summary:")
        print(f"  Fields processed: {summary['fields_processed']}")
        print(f"  Method: {summary['anonymization_method']}")
        print(f"  Fields changed: {sum(1 for details in summary['field_details'].values() if details['changed'])}")
        
        # Demonstrate differential privacy
        print("\n🔐 Differential Privacy Demo:")
        original_value = 1000.0
        noisy_values = [self.anonymizer.add_differential_privacy_noise(original_value) for _ in range(5)]
        print(f"  Original value: {original_value}")
        print(f"  Noisy values: {[round(v, 2) for v in noisy_values]}")
    
    def demo_consent_management(self):
        """
        Demonstrate consent management capabilities
        """
        print("\n" + "="*60)
        print("📝 CONSENT MANAGEMENT DEMONSTRATION")
        print("="*60)
        
        user_id = "user_12345"
        
        # Create consent requests
        print(f"👤 Creating consent requests for user: {user_id}")
        
        fraud_detection_consent = self.consent_manager.create_consent_request(
            user_id=user_id,
            template_id='fraud_detection',
            metadata={'source': 'mobile_app', 'version': '2.1.0'}
        )
        
        analytics_consent = self.consent_manager.create_consent_request(
            user_id=user_id,
            template_id='analytics',
            metadata={'source': 'web_app'}
        )
        
        marketing_consent = self.consent_manager.create_consent_request(
            user_id=user_id,
            template_id='marketing',
            metadata={'source': 'onboarding'}
        )
        
        print(f"  Created consent IDs: {fraud_detection_consent}, {analytics_consent}, {marketing_consent}")
        
        # Grant some consents
        print("\n✅ Granting consents:")
        self.consent_manager.grant_consent(fraud_detection_consent, user_id)
        self.consent_manager.grant_consent(analytics_consent, user_id)
        # Marketing consent remains pending
        
        print("  - Fraud detection: GRANTED")
        print("  - Analytics: GRANTED")
        print("  - Marketing: PENDING")
        
        # Check consent status
        print("\n🔍 Checking consent status:")
        has_fraud_consent = self.consent_manager.check_consent(
            user_id, ConsentPurpose.LEGITIMATE_INTERESTS, 'transaction_data'
        )
        has_marketing_consent = self.consent_manager.check_consent(
            user_id, ConsentPurpose.CONSENT, 'contact_data'
        )
        
        print(f"  Fraud detection consent: {has_fraud_consent}")
        print(f"  Marketing consent: {has_marketing_consent}")
        
        # Get consent summary
        summary = self.consent_manager.get_consent_summary(user_id)
        print(f"\n📊 Consent Summary:")
        print(f"  Total consents: {summary['total_consents']}")
        print(f"  Active consents: {summary['active_consents']}")
        print(f"  Data categories: {', '.join(summary['data_categories'])}")
        
        # Withdraw consent demo
        print("\n🚫 Withdrawing analytics consent:")
        self.consent_manager.withdraw_consent(analytics_consent, user_id, "User requested data minimization")
        print("  Analytics consent withdrawn")
        
        # Updated status
        updated_summary = self.consent_manager.get_consent_summary(user_id)
        print(f"  Updated active consents: {updated_summary['active_consents']}")
    
    def demo_gdpr_compliance(self):
        """
        Demonstrate GDPR compliance capabilities
        """
        print("\n" + "="*60)
        print("⚖️ GDPR COMPLIANCE DEMONSTRATION")
        print("="*60)
        
        user_id = "user_67890"
        
        # Create data subject requests
        print("📋 Creating GDPR data subject requests:")
        
        # Right of access request
        access_request = self.gdpr_manager.create_data_subject_request(
            user_id=user_id,
            right=GDPRRight.RIGHT_OF_ACCESS,
            details="User wants to know what data we have about them"
        )
        print(f"  Access request: {access_request}")
        
        # Right to erasure request
        erasure_request = self.gdpr_manager.create_data_subject_request(
            user_id=user_id,
            right=GDPRRight.RIGHT_OF_ERASURE,
            details="User wants all their data deleted"
        )
        print(f"  Erasure request: {erasure_request}")
        
        # Process access request
        print("\n🔍 Processing access request:")
        sample_user_data = {
            'personal_info': {
                'name': 'Jane Smith',
                'email': 'jane.smith@example.com',
                'phone': '+1-555-987-6543'
            },
            'account_info': {
                'account_id': 'ACC_123456',
                'created_date': '2023-01-15',
                'last_login': '2024-12-10'
            },
            'transaction_history': [
                {'date': '2024-12-01', 'amount': 150.00, 'type': 'purchase'},
                {'date': '2024-12-05', 'amount': 75.50, 'type': 'transfer'}
            ]
        }
        
        access_processed = self.gdpr_manager.process_access_request(access_request, sample_user_data)
        print(f"  Access request processed: {access_processed}")
        
        # Create DPIA
        print("\n📋 Creating Data Protection Impact Assessment:")
        dpia_id = self.gdpr_manager.create_dpia(
            system_name="AI Fraud Detection System",
            processing_purpose="Real-time fraud detection using machine learning",
            risk_level="high",
            reviewer="Data Protection Officer"
        )
        print(f"  DPIA created: {dpia_id}")
        
        # Update DPIA findings
        self.gdpr_manager.update_dpia_findings(
            dpia_id,
            findings=[
                "System processes large volumes of personal data",
                "Automated decision making has significant impact on individuals",
                "Data sharing with third parties increases risk exposure"
            ],
            recommendations=[
                "Implement privacy by design principles",
                "Add human review for high-impact decisions",
                "Enhance data encryption and access controls"
            ],
            mitigation_measures=[
                "Role-based access control implemented",
                "Data minimization techniques applied",
                "Regular privacy impact assessments scheduled"
            ]
        )
        
        # Approve DPIA
        self.gdpr_manager.approve_dpia(dpia_id, "Senior Compliance Officer")
        print("  DPIA findings updated and approved")
        
        # Report data breach
        print("\n🚨 Reporting data breach:")
        breach_id = self.gdpr_manager.report_data_breach(
            severity=BreachSeverity.HIGH,
            affected_users=1500,
            data_types=['email_addresses', 'phone_numbers', 'transaction_data'],
            description="Unauthorized access to customer database due to misconfigured API"
        )
        print(f"  Breach reported: {breach_id}")
        
        # Contain breach
        self.gdpr_manager.contain_breach(
            breach_id,
            mitigation_actions=[
                "Secured vulnerable API endpoint",
                "Revoked compromised access tokens",
                "Notified affected users",
                "Reported to supervisory authority"
            ]
        )
        print("  Breach contained with mitigation actions")
        
        # Generate compliance report
        print("\n📊 Generating compliance report:")
        compliance_report = self.gdpr_manager.generate_compliance_report()
        print(f"  Total data subject requests: {compliance_report['data_subject_requests']['total']}")
        print(f"  DPIA assessments: {compliance_report['dpia_assessments']['total']}")
        print(f"  Data breaches: {compliance_report['data_breaches']['total']}")
        print(f"  Processing activities: {compliance_report['processing_activities']}")
    
    def demo_access_control(self):
        """
        Demonstrate access control capabilities
        """
        print("\n" + "="*60)
        print("🔐 ACCESS CONTROL DEMONSTRATION")
        print("="*60)
        
        # Test different access scenarios
        test_scenarios = [
            {
                'user_id': 'analyst_001',
                'role': UserRole.ANALYST,
                'resource': 'fraud_cases',
                'operation': 'read',
                'purpose': AccessPurpose.FRAUD_INVESTIGATION,
                'sensitivity': DataSensitivity.CONFIDENTIAL,
                'context': {'ip_address': '192.168.1.50', 'location': 'US'}
            },
            {
                'user_id': 'compliance_001',
                'role': UserRole.COMPLIANCE_OFFICER,
                'resource': 'audit_logs',
                'operation': 'export',
                'purpose': AccessPurpose.COMPLIENCE_AUDIT,
                'sensitivity': DataSensitivity.RESTRICTED,
                'context': {'ip_address': '192.168.1.60', 'location': 'US'}
            },
            {
                'user_id': 'analyst_001',
                'role': UserRole.ANALYST,
                'resource': 'customer_pii',
                'operation': 'read',
                'purpose': AccessPurpose.DATA_ANALYSIS,
                'sensitivity': DataSensitivity.CRITICAL,
                'context': {'ip_address': '192.168.1.50', 'location': 'US'}
            }
        ]
        
        print("🔍 Testing access control scenarios:")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n  Scenario {i}: {scenario['role'].value} accessing {scenario['resource']}")
            
            # Check access
            granted, reason = self.access_controller.check_access(
                scenario['user_id'],
                scenario['role'],
                scenario['resource'],
                scenario['operation'],
                scenario['purpose'],
                scenario['sensitivity'],
                scenario['context']
            )
            
            print(f"    Access granted: {granted}")
            if not granted:
                print(f"    Reason: {reason}")
            
            # If access requires approval, create request
            if not granted and "Approval required" in reason:
                request_id = self.access_controller.request_access(
                    scenario['user_id'],
                    scenario['role'],
                    scenario['resource'],
                    scenario['operation'],
                    scenario['purpose'],
                    scenario['sensitivity'],
                    "Need access for investigation",
                    expires_hours=24
                )
                print(f"    Created access request: {request_id}")
                
                # Approve the request
                self.access_controller.approve_request(request_id, "supervisor_001")
                print(f"    Request approved by supervisor")
        
        # Get access statistics
        print("\n📊 Access Control Statistics:")
        stats = self.access_controller.get_access_statistics()
        print(f"  Total access attempts: {stats['total_access_attempts']}")
        print(f"  Granted access: {stats['granted_access']}")
        print(f"  Denied access: {stats['denied_access']}")
        print(f"  Approval rate: {stats['approval_rate']:.2%}")
        print(f"  Pending requests: {stats['pending_requests']}")
        print(f"  Active policies: {stats['active_policies']}")
    
    def demo_data_classification(self):
        """
        Demonstrate data classification capabilities
        """
        print("\n" + "="*60)
        print("🏷️ DATA CLASSIFICATION DEMONSTRATION")
        print("="*60)
        
        # Sample data to classify
        test_data = [
            "john.doe@example.com",
            "+1-555-123-4567",
            "123-45-6789",
            "4532-1234-5678-9012",
            "192.168.1.100",
            "$1,250.75",
            "123 Main St, New York, NY 10001",
            "User logged in from Chrome browser",
            "Transaction approved for $500.00"
        ]
        
        print("🔍 Classifying sample data:")
        
        classifications = []
        for i, data in enumerate(test_data):
            classification = self.data_classifier.classify_data(data, f"test_data_{i}")
            classifications.append(classification)
            
            print(f"  Data: '{data}'")
            print(f"    Category: {classification.category.value}")
            print(f"    Sensitivity: {classification.sensitivity.value}")
            print(f"    Confidence: {classification.confidence:.2f}")
            print(f"    Regulatory scope: {[scope.value for scope in classification.regulatory_scope]}")
            print()
        
        # Classify a complete record
        print("📋 Classifying complete customer record:")
        customer_record = {
            'customer_id': 'CUST_001',
            'name': 'Alice Johnson',
            'email': 'alice.johnson@example.com',
            'phone': '+1-555-987-6543',
            'ssn': '987-65-4321',
            'credit_card': '5555-1234-5678-9012',
            'ip_address': '10.0.0.15',
            'transaction_amount': 750.25,
            'address': '456 Oak Ave, Los Angeles, CA 90001'
        }
        
        record_classifications = self.data_classifier.classify_record(customer_record)
        
        for field, classification in record_classifications.items():
            print(f"  {field}: {classification.sensitivity.value} ({classification.category.value})")
        
        # Get classification statistics
        print("\n📊 Classification Statistics:")
        stats = self.data_classifier.get_classification_statistics()
        print(f"  Total classifications: {stats['total_classifications']}")
        print(f"  Active rules: {stats['active_rules']}")
        print(f"  Average confidence: {stats['average_confidence']:.2f}")
        
        print("\n  Sensitivity distribution:")
        for sensitivity, count in stats['sensitivity_distribution'].items():
            print(f"    {sensitivity}: {count}")
        
        print("\n  Category distribution:")
        for category, count in stats['category_distribution'].items():
            print(f"    {category}: {count}")
        
        # PII detection demo
        print("\n🔍 PII Detection Demo:")
        sample_text = "Contact john.doe@example.com or call +1-555-123-4567. SSN: 123-45-6789"
        pii_types = self.data_classifier.detect_pii_types(sample_text)
        print(f"  Text: {sample_text}")
        print(f"  Detected PII types: {pii_types}")
    
    def run_complete_demo(self):
        """
        Run the complete privacy framework demonstration
        """
        print("🚀 STARTING COMPLETE PRIVACY CONTROL FRAMEWORK DEMO")
        print("="*80)
        
        try:
            # Run all demonstrations
            self.demo_data_anonymization()
            self.demo_consent_management()
            self.demo_gdpr_compliance()
            self.demo_access_control()
            self.demo_data_classification()
            
            print("\n" + "="*80)
            print("✅ PRIVACY CONTROL FRAMEWORK DEMO COMPLETED SUCCESSFULLY!")
            print("="*80)
            
            print("\n🎯 Key Features Demonstrated:")
            print("  🔐 Data Anonymization & Pseudonymization")
            print("  📝 Consent Management & Tracking")
            print("  ⚖️ GDPR Compliance Management")
            print("  🛡️ Access Control & Authorization")
            print("  🏷️ Data Classification & Sensitivity Analysis")
            
            print("\n🔧 Privacy Controls Implemented:")
            print("  • Field-level anonymization with format preservation")
            print("  • Granular consent management with withdrawal support")
            print("  • GDPR Articles 15, 16, 17, 20, 21 compliance")
            print("  • Role-based access control with approval workflows")
            print("  • Automated PII detection and classification")
            print("  • Differential privacy for statistical analysis")
            print("  • Data breach notification and management")
            print("  • DPIA (Data Protection Impact Assessment) support")
            
            print("\n📊 Compliance Frameworks Supported:")
            print("  • GDPR (General Data Protection Regulation)")
            print("  • CCPA (California Consumer Privacy Act)")
            print("  • PCI DSS (Payment Card Industry Data Security Standard)")
            print("  • HIPAA (Health Insurance Portability and Accountability Act)")
            
        except Exception as e:
            print(f"\n❌ Demo failed with error: {str(e)}")
            import traceback
            traceback.print_exc()

def main():
    """
    Main function to run the privacy framework demo
    """
    demo = PrivacyFrameworkDemo()
    demo.run_complete_demo()

if __name__ == "__main__":
    main()