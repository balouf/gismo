# Skip spacy-dependent modules when spacy is not available (e.g., Python 3.14)
collect_ignore = []

try:
    import spacy
except ImportError:
    collect_ignore.append("gismo/sentencizer.py")
