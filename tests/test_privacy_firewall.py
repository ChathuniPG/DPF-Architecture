"""
Validation Suite: Deterministic Privacy Firewall
------------------------------------------------
Unit testing framework for the Privacy Sanitization Engine.
Validates the correctness of regex-based redaction against
adversarial inputs, edge cases, and known threat vectors.

Scope:
1. Verification of Medical/Health pattern detection.
2. Verification of Financial/Pecuniary data redaction.
3. Verification of Academic Performance indicators.
4. False Positive Analysis (Control Group).
"""

import unittest
import sys
import os

# --- ENVIRONMENT SETUP ---
# Adjust python path to include the 'src' directory for module resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Import from dpf_core (Robust import handling)
try:
    from dpf_core.privacy_firewall import PrivacyFirewall
except ImportError:
    # Fallback if python doesn't resolve the package structure immediately
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'dpf_core'))
    from privacy_firewall import PrivacyFirewall

class TestPrivacyFirewall(unittest.TestCase):
    
    def setUp(self):
        """
        Test Harness Initialization.
        Instantiates the optimized firewall engine before each test case.
        """
        self.fw = PrivacyFirewall()

    def assertRedacted(self, text, expected_tag):
        """
        Assertion Helper: Validates that a specific privacy trigger fires.
        
        Args:
            text (str): The adversarial input vector.
            expected_tag (str): The redaction token expected in the output.
        """
        result = self.fw.evaluate(text)
        
        # 1. Verify Trigger Flag
        self.assertTrue(result.is_triggered, 
            f" [FAILURE] Security mechanism failed to trigger on input: '{text}'")
        
        # 2. Verify Sanitization
        self.assertIn(expected_tag, result.clean_text, 
            f" [FAILURE] Expected token {expected_tag} missing from output: '{result.clean_text}'")

    def assertAllowed(self, text):
        """
        Assertion Helper: Validates that benign text passes unmodified (Utility Check).
        """
        result = self.fw.evaluate(text)
        
        # 1. Verify False Positive Rate
        self.assertFalse(result.is_triggered, 
            f" [FAILURE] False Positive detected on safe input: '{text}'")
        
        # 2. Verify Integrity
        self.assertEqual(text, result.clean_text,
            f" [FAILURE] Content mutation occurred on safe input.")

    # ==========================================
    # TEST VECTOR A: SENSITIVE ATTRIBUTES (Medical)
    # ==========================================
    def test_medical_incident_detection(self):
        """Validates detection of acute health incidents (e.g., Panic Attacks)."""
        vectors = [
            "I am having a panic attack right now.",
            "She had a Panic Attack during the exam.",
            "He is suffering from severe PANIC ATTACKS."
        ]
        for v in vectors:
            self.assertRedacted(v, "[HEALTH_INCIDENT]")

    def test_medical_trigger_patterns(self):
        """Validates detection of physiological triggers (e.g., Fainting/Blood)."""
        vectors = [
            "I feel faint when I see blood.",
            "I might FAINT if there is too much blood.",
            "faint... blood everywhere."
        ]
        for v in vectors:
            self.assertRedacted(v, "[MEDICAL_TRIGGER]")

    def test_medical_condition_identification(self):
        """Validates detection of chronic conditions (e.g., Migraines)."""
        vectors = [
            "I have a migraine.",
            "These migraines are killing me.",
            "Migraine aura detected."
        ]
        for v in vectors:
            self.assertRedacted(v, "[HEALTH_CONDITION]")

    # ==========================================
    # TEST VECTOR B: FINANCIAL DATA (Pecuniary)
    # ==========================================
    def test_currency_format_detection(self):
        """Validates detection of standard currency symbols and amounts."""
        vectors = [
            "I owe $5000 to the school.",
            "The fee is $10.",
            "It costs $1,000,000."
        ]
        for v in vectors:
            self.assertRedacted(v, "[FINANCIAL_ALERT]")

    def test_natural_language_currency(self):
        """Validates detection of currency expressed in natural language."""
        vectors = [
            "It costs 500 dollars.",
            "I need 10 dollars for lunch.",
            "About 5000 DOLLARS debt."
        ]
        for v in vectors:
            self.assertRedacted(v, "[FINANCIAL_ALERT]")

    # ==========================================
    # TEST VECTOR C: ACADEMIC PERFORMANCE (Grades)
    # ==========================================
    def test_academic_grade_syntax(self):
        """Validates detection of explicit grade reporting formats."""
        vectors = [
            "My Grade is F.",
            "Grade: F",
            "Grade          F", # Adversarial spacing
            "My current grade F is bad."
        ]
        for v in vectors:
            self.assertRedacted(v, "[ACADEMIC_ALERT]")

    def test_contextual_grade_inference(self):
        """Validates detection of grades inferred from context sentences."""
        vectors = [
            "The result is an F.",
            "It is an F for this semester.",
            "I got an F in history."
        ]
        for v in vectors:
            self.assertRedacted(v, "[ACADEMIC_ALERT]")

    def test_explicit_failure_status(self):
        """Validates detection of explicit 'Fail' markers."""
        vectors = [
            "Grade: F (Fail)",
            "I got an F (Fail) on the test.",
            "Status: F(Fail)"
        ]
        for v in vectors:
            self.assertRedacted(v, "[ACADEMIC_ALERT]")

    def test_performance_verbs(self):
        """Validates detection of performance-related verbs."""
        vectors = [
            "I am failing this class.",
            "I might FAIL history.",
            "Why am I failing?"
        ]
        for v in vectors:
            self.assertRedacted(v, "[PERFORMANCE_FLAG]")

    def test_low_score_indicators(self):
        """Validates detection of specific low-grade patterns."""
        vectors = [
            "I got a C- on the quiz.",
            "My grade is a D+.",
            "I have a D average."
        ]
        for v in vectors:
            self.assertRedacted(v, "[ACADEMIC_WARNING]")

    # ==========================================
    # TEST VECTOR D: ACADEMIC STANDING (Status)
    # ==========================================
    def test_academic_status_markers(self):
        """Validates detection of disciplinary or probational status."""
        vectors = [
            "I am on academic probation.",
            "I am on probation.",
            "I face suspension from school.",
            "I received an academic warning.",
            "I am in poor standing."
        ]
        for v in vectors:
            self.assertRedacted(v, "[ACADEMIC_STATUS_REDACTED]")

    # ==========================================
    # CONTROL GROUP: UTILITY PRESERVATION
    # ==========================================
    def test_false_positive_control(self):
        """
        Validates that benign, domain-relevant text is preserved.
        Ensures the firewall does not degrade system utility.
        """
        safe_vectors = [
            "I like history class.",
            "What is the capital of France?",
            "I feel happy today.",
            "I am planning a schedule.",
            "Is the deadline tomorrow?", # Trigger for Logic Agent, but Safe for Privacy
            "This is a fun game."
        ]
        for v in safe_vectors:
            self.assertAllowed(v)

if __name__ == "__main__":
    print("==========================================================")
    print("    DPF FIREWALL: FORMAL VERIFICATION SUITE               ")
    print("==========================================================")
    unittest.main()