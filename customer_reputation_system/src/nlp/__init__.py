"""NLP module for text analysis and classification"""

# Lazy imports to avoid circular dependencies
__all__ = ["TextAnalyzer", "ReportTextClassifier", "SimpleTextClassifier"]

def __getattr__(name):
    if name == "TextAnalyzer":
        from maifds_governance.nlp.text_analyzer import TextAnalyzer
        return TextAnalyzer
    elif name == "ReportTextClassifier":
        from maifds_governance.nlp.model import ReportTextClassifier
        return ReportTextClassifier
    elif name == "SimpleTextClassifier":
        from maifds_governance.nlp.model import SimpleTextClassifier
        return SimpleTextClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

