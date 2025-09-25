#!/usr/bin/env python3
"""
NLP Preprocessing Examples
A minimal runnable script demonstrating basic NLP preprocessing techniques.
"""

import nltk
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

def main():
    # Sample text
    text = """Natural language processing (NLP) is a subfield of linguistics, computer science, 
    and artificial intelligence concerned with the interactions between computers and human language. 
    The goal is a computer capable of understanding the contents of documents."""
    
    print("Original text:")
    print(text)
    print("\n" + "="*50 + "\n")
    
    # 1. Tokenization with NLTK
    print("1. Tokenization (NLTK):")
    tokens = word_tokenize(text)
    print(tokens[:15], "...\n")
    
    # 2. Stopword Removal
    print("2. Stopword Removal:")
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    print(filtered_tokens[:15], "...\n")
    
    # 3. Stemming
    print("3. Stemming (Porter):")
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    print(stemmed_tokens[:15], "...\n")
    
    # 4. Lemmatization
    print("4. Lemmatization (WordNet):")
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    print(lemmatized_tokens[:15], "...\n")
    
    # 5. POS Tagging
    print("5. Part-of-Speech Tagging:")
    pos_tags = nltk.pos_tag(tokens[:15])
    print(pos_tags, "...\n")
    
    # 6. spaCy processing
    print("6. spaCy Processing:")
    try:
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        
        # Named Entity Recognition
        print("Named Entities:")
        for ent in doc.ents:
            print(f"  - {ent.text} ({ent.label_})")
        
        # spaCy tokens and lemmas
        print("\nspaCy tokens and lemmas:")
        for token in list(doc)[:10]:
            print(f"  {token.text} -> {token.lemma_} (POS: {token.pos_})")
            
    except OSError:
        print("spaCy model not found. Please install it with:")
        print("python -m spacy download en_core_web_sm")

if __name__ == "__main__":
    print("NLP Preprocessing Examples")
    print("="*30)
    main()
    print("\nTo run more comprehensive examples, check the notebook.ipynb file.")