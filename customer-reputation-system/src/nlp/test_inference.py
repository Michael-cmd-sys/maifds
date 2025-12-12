"""
Test script for NLP text analysis inference
"""

from src.nlp.text_analyzer import TextAnalyzer
from config.logging_config import setup_logger

logger = setup_logger(__name__)


def test_analysis():
    """Test NLP text analysis with sample texts"""
    
    analyzer = TextAnalyzer()
    
    # Sample report texts
    test_cases = [
        {
            "title": "Unauthorized charge on my account",
            "description": "I noticed a charge of $150 that I did not authorize. The merchant charged me twice for the same transaction without authorization. This is fraud!"
        },
        {
            "title": "Poor customer service",
            "description": "The merchant was unresponsive to my inquiries about delivery status. It took over 3 weeks to receive my order. Very disappointed."
        },
        {
            "title": "Payment processing error",
            "description": "Encountered multiple errors while trying to complete payment. Had to try 5 times before it went through. Technical issue."
        },
        {
            "title": "Great service",
            "description": "The product quality is excellent, and shipping was fast. Very satisfied with the purchase. Highly recommend!"
        },
    ]
    
    print("=" * 80)
    print("NLP Text Analysis Test")
    print("=" * 80)
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Title: {test_case['title']}")
        print(f"  Description: {test_case['description'][:100]}...")
        print()
        
        analysis = analyzer.analyze_report(
            title=test_case['title'],
            description=test_case['description']
        )
        
        print("  Analysis Results:")
        print(f"    Sentiment: {analysis['sentiment']} (confidence: {analysis['sentiment_confidence']:.2f})")
        print(f"    Urgency: {analysis['urgency']} (confidence: {analysis['urgency_confidence']:.2f})")
        print(f"    Credibility Score: {analysis['credibility_score']:.2f}")
        print()
        
        if 'text_features' in analysis:
            features = analysis['text_features']
            print("  Text Features:")
            print(f"    Word Count: {features.get('word_count', 0)}")
            print(f"    Urgency Keywords: {features.get('urgency_keyword_count', 0)}")
            print(f"    Negative Words: {features.get('negative_word_count', 0)}")
            print()
        
        print("-" * 80)
        print()


if __name__ == "__main__":
    test_analysis()

