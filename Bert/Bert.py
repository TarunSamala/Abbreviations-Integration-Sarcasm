import json
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification, DistilBertConfig

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

# Configuration
ABBREV_PATH = '../Dataset/Abbreviations.csv'  # Add abbreviation path
OUTPUT_DIR = 'sarcasm_outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------- Abbreviation Handling ---------------------- #
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
            abbreviation_dict[core_abbrev] = re.sub(r'\s+', ' ', clean_meaning)
            
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

# ---------------------- Data Loading ---------------------- #
def load_data(file_path, abbreviation_dict):
    """Load and preprocess dataset with abbreviations"""
    with open(file_path, 'r') as f:
        datastore = [json.loads(line) for line in f]
    
    df = pd.DataFrame(datastore)
    df = df[['is_sarcastic', 'headline']]
    df['clean_headline'] = df['headline'].apply(lambda x: clean_text(x, abbreviation_dict))
    return df

# ---------------------- Main Execution ---------------------- #
if __name__ == "__main__":
    # Load abbreviations first
    abbrev_dict = load_abbreviations(ABBREV_PATH)
    
    # Load and split data
    df = load_data('../Dataset/Sarcasm_Headlines_Dataset_v2.json', abbrev_dict)
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_headline'],
        df['is_sarcastic'],
        test_size=0.2,
        random_state=42,
        stratify=df['is_sarcastic']
    )

    # Class weights with smoothing
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weights = np.clip(class_weights, 0.5, 2)
    class_weights_dict = dict(zip(classes, class_weights))
    sample_weights = np.array([class_weights_dict[label] for label in y_train])

    # Tokenization with dynamic padding
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    train_encodings = tokenizer(
        X_train.tolist(),
        truncation=True,
        padding=True,
        max_length=40,
        return_tensors='tf'
    )
    test_encodings = tokenizer(
        X_test.tolist(),
        truncation=True,
        padding=True,
        max_length=40,
        return_tensors='tf'
    )

    # Optimized dataset preparation
    train_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_encodings['input_ids'], 
         'attention_mask': train_encodings['attention_mask']},
        y_train,
        sample_weights
    )).shuffle(1000).batch(16).prefetch(tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        {'input_ids': test_encodings['input_ids'],
         'attention_mask': test_encodings['attention_mask']},
        y_test
    )).batch(16).prefetch(tf.data.AUTOTUNE)

    # Enhanced model configuration
    config = DistilBertConfig.from_pretrained(
        'distilbert-base-uncased',
        num_labels=1,
        dropout=0.4,
        seq_classif_dropout=0.5,
        attention_dropout=0.2
    )
    
    model = TFDistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', config=config)
    
    # Enhanced regularization
    l2_reg = tf.keras.regularizers.l2(0.02)
    model.classifier = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=model.classifier.kernel_initializer,
            kernel_regularizer=l2_reg
        )
    ])

    # Optimized learning schedule
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=1e-5,
        weight_decay=0.03,
        clipnorm=1.0
    )
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Enhanced callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=0.005,
        restore_best_weights=True
    )
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1,
        min_lr=1e-6
    )

    # Training
    history = model.fit(
        train_dataset,
        epochs=4,
        validation_data=test_dataset,
        callbacks=[early_stop, lr_scheduler],
        verbose=1
    )

    # Visualization and reporting
    def save_training_curves(history):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train')
        plt.plot(history.history['val_accuracy'], label='Validation')
        plt.title('Accuracy Curves')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Validation')
        plt.title('Loss Curves')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(OUTPUT_DIR, 'training_curves.png'), bbox_inches='tight')
        plt.close()

    save_training_curves(history)

    logits = model.predict(test_dataset).logits
    probabilities = tf.sigmoid(logits).numpy().flatten()
    y_pred = (probabilities > 0.5).astype(int)

    report = classification_report(y_test, y_pred)
    with open(os.path.join(OUTPUT_DIR, 'classification_report.txt'), 'w') as f:
        f.write("Classification Report:\n")
        f.write(report)

    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Non-Sarcastic', 'Sarcastic'],
                yticklabels=['Non-Sarcastic', 'Sarcastic'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), bbox_inches='tight')
    plt.close()

    print("All results saved to 'sarcasm_outputs' directory")