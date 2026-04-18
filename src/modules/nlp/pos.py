import spacy
nlp = spacy.load("en_core_web_sm")

class POS:
    """
    Docstring for POS: 
    Tokenize the question splits into words
    punctuation and part of speech tagging
    """
    def __init__(self):
        pass
    def doc(self, question):
        doc = nlp(question)
        return doc