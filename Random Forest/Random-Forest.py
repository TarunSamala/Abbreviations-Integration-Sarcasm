import json
import pandas as pd
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
DATA_PATH = '../Dataset/Sarcasm_Headlines_Dataset_v2.json'
ABBREV_PATH = '../Dataset/Abbreviations.csv'  # New abbreviation path
OUTPUT_DIR = "sarcasm_outputs"

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_abbreviations(abbrev_file):
    """Load and process abbreviation dictionary"""
    abbrev_df = pd.read_csv(abbrev_file)
    abbreviation_dict = {}
    
    # Normalize column names
    abbrev_df.columns = [col.strip().lower() for col in abbrev_df.columns]
    
    for _, row in abbrev_df.iterrows():
        abbrev = str(row['word']).strip()
        meaning = str(row['meaning']).strip()
        
        # Create core version of abbreviation
        core_abbrev = re.sub(r'[^a-z0-9]', '', abbrev.lower())
        clean_meaning = re.sub(r'[^a-zA-Z\s]', '', meaning.lower()).strip()
        
        if core_abbrev and clean_meaning:
            abbreviation_dict[core_abbrev] = clean_meaning
            
    return abbreviation_dict

def clean_text(text, abbreviation_dict):
    """Enhanced cleaning with abbreviation expansion"""
    text = str(text).lower()
    words = text.split()
    processed_words = []
    
    for word in words:
        # Extract core form (alphanumeric only)
        core = re.sub(r'[^a-z0-9]', '', word)
        if core in abbreviation_dict:
            processed_words.append(abbreviation_dict[core])
        else:
            processed_words.append(word)
    
    text = ' '.join(processed_words)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def load_data(file_path, abbreviation_dict):
    """Load and preprocess dataset with abbreviations"""
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(lambda x: clean_text(x, abbreviation_dict))
    return df

# Main execution
if __name__ == "__main__":
    # Load abbreviations first
    abbrev_dict = load_abbreviations(ABBREV_PATH)
    
    # Load and split data
    df = load_data(DATA_PATH, abbrev_dict)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'], 
        df['is_sarcastic'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['is_sarcastic']
    )

    # TF-IDF Vectorization
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, 
                               class_weight='balanced',
                               n_jobs=-1,
                               random_state=42)
    rf.fit(X_train_tfidf, y_train)

    # Generate predictions
    y_pred = rf.predict(X_test_tfidf)

    # Save classification report
    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'classification_report-rf.txt'), 'w') as f:
        f.write("Random Forest Classification Report:\n")
        f.write(report)

    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Random Forest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix-rf.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print("Results saved to:", OUTPUT_DIR)