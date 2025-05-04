import os
import re
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Embedding, Bidirectional, GRU, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ---------------------- Configuration ---------------------- #
MAX_LEN = 35
VOCAB_SIZE = 12000
EMBEDDING_DIM = 96
BATCH_SIZE = 128
EPOCHS = 40
OUTPUT_DIR = "bigru_sarcasm_outputs"
DATA_PATH = "../Dataset/Sarcasm_Headlines_Dataset_v2.json"
ABBREV_PATH = "../Dataset/Abbreviations.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Abbreviation Handling ---------------------- #
def load_abbreviations(abbrev_file):
    abbrev_df = pd.read_csv(abbrev_file)
    abbreviation_dict = {}
    abbrev_df.columns = [col.strip().lower() for col in abbrev_df.columns]
    
    for _, row in abbrev_df.iterrows():
        abbrev = str(row['word']).strip()
        meaning = str(row['meaning']).strip()
        core_abbrev = re.sub(r'[^a-z0-9]', '', abbrev.lower())
        clean_meaning = re.sub(r'[^a-zA-Z\s]', '', meaning.lower()).strip()
        
        if core_abbrev and clean_meaning:
            abbreviation_dict[core_abbrev] = re.sub(r'\s+', ' ', clean_meaning)
                
    return abbreviation_dict

def clean_text(text, abbreviation_dict):
    text = str(text).lower()
    words = text.split()
    processed_words = []
    
    for word in words:
        core = re.sub(r'[^a-z0-9]', '', word)
        processed_words.append(abbreviation_dict.get(core, word))
    
    text = ' '.join(processed_words)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

# ---------------------- Data Loading ---------------------- #
def load_data(file_path, abbreviation_dict):
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    texts, labels = [], []
    for entry in datastore:
        raw_text = entry.get("headline", "")
        cleaned = clean_text(raw_text, abbreviation_dict)
        texts.append(cleaned)
        labels.append(entry.get("is_sarcastic", 0))
    
    return texts, np.array(labels)

# ---------------------- Model Architecture ---------------------- #
def build_bigru_model():
    inputs = Input(shape=(MAX_LEN,))
    
    x = Embedding(VOCAB_SIZE, EMBEDDING_DIM, 
                embeddings_regularizer=regularizers.l2(1e-4))(inputs)
    x = SpatialDropout1D(0.6)(x)
    
    x = Bidirectional(GRU(64,
                        activation='tanh',
                        kernel_regularizer=regularizers.l2(1e-4),
                        recurrent_dropout=0.3,
                        dropout=0.5))(x)
    
    x = Dense(96, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.7)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Nadam(learning_rate=2e-4, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ---------------------- Training Setup ---------------------- #
abbreviation_dict = load_abbreviations(ABBREV_PATH)
texts, labels = load_data(DATA_PATH, abbreviation_dict)

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>', filters='')
tokenizer.fit_on_texts(texts)
X = pad_sequences(tokenizer.texts_to_sequences(texts), maxlen=MAX_LEN, padding='post')

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=7, min_delta=0.001, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)
]

# ---------------------- Training Execution ---------------------- #
model = build_bigru_model()
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    class_weight={0: 1.25, 1: 0.8},
    verbose=1
)

# ---------------------- Visualization ---------------------- #
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Curves')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss Curves')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.grid(linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), dpi=300)
plt.close()

# ---------------------- Evaluation ---------------------- #
y_pred = (model.predict(X_test) > 0.5).astype(int)

# Classification Report
report = classification_report(y_test, y_pred, target_names=['Not Sarcastic', 'Sarcastic'])
with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
    f.write("Classification Report:\n")
    f.write(report)

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Not Sarcastic', 'Sarcastic'],
            yticklabels=['Not Sarcastic', 'Sarcastic'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
plt.close()

print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")