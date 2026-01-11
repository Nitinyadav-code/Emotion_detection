import json

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Emotion Detection from Text using LSTM\n",
    "\n",
    "## 1. Introduction\n",
    "**Objective**: Build a deep learning model to classify the emotion of a given text message.\n",
    "\n",
    "**Dataset**: We will use the [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset (formerly typical of Twitter data). It contains English Twitter messages labeled with six emotions:\n",
    "- Anger\n",
    "- Fear\n",
    "- Joy\n",
    "- Love\n",
    "- Sadness\n",
    "- Surprise\n",
    "\n",
    "**Architecture**: \n",
    "- **Embedding Layer**: Converts words into dense vectors.\n",
    "- **Bidirectional LSTM Layers**: Captures context from both directions (past and future).\n",
    "- **Dense Layers**: For classification."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Setup and Imports\n",
    "Checking for GPU availability and installing necessary libraries."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install Hugging Face datasets library if not already installed\n",
    "!pip install datasets seaborn matplotlib scikit-learn tensorflow wget --quiet\n",
    "\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import tensorflow as tf\n",
    "from datasets import load_dataset\n",
    "from tensorflow.keras.preprocessing.text import Tokenizer\n",
    "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
    "from tensorflow.keras.models import Sequential\n",
    "from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Bidirectional, Dropout, BatchNormalization\n",
    "from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "from sklearn.utils.class_weight import compute_class_weight\n",
    "import os\n",
    "import wget\n",
    "\n",
    "# Set seed for reproducibility\n",
    "tf.random.set_seed(42)\n",
    "np.random.seed(42)\n",
    "\n",
    "print(\"TensorFlow Version:\", tf.__version__)\n",
    "print(\"GPU Available:\", \"Yes\" if tf.config.list_physical_devices('GPU') else \"No\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Loading\n",
    "We use the `datasets` library to easily load the emotion dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset\n",
    "dataset = load_dataset(\"dair-ai/emotion\")\n",
    "\n",
    "# View dataset structure\n",
    "print(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert to Pandas DataFrames for easier manipulation\n",
    "train_df = pd.DataFrame(dataset['train'])\n",
    "val_df = pd.DataFrame(dataset['validation'])\n",
    "test_df = pd.DataFrame(dataset['test'])\n",
    "\n",
    "# Check first few rows\n",
    "print(train_df.head())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define Label Mapping (Dynamic)\n",
    "labels = dataset['train'].features['label'].names\n",
    "label_map = {i: label for i, label in enumerate(labels)}\n",
    "print(\"Label Mapping:\", label_map)\n",
    "\n",
    "# Add a 'label_name' column for readability\n",
    "train_df['label_name'] = train_df['label'].map(label_map)\n",
    "val_df['label_name'] = val_df['label'].map(label_map)\n",
    "test_df['label_name'] = test_df['label'].map(label_map)\n",
    "\n",
    "train_df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Exploratory Data Analysis (EDA)\n",
    "Understanding the distribution of emotions and text length."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Class Distribution\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.countplot(data=train_df, x='label_name', order=train_df['label_name'].value_counts().index, palette='viridis')\n",
    "plt.title('Distribution of Emotions in Training Data')\n",
    "plt.xlabel('Emotion')\n",
    "plt.ylabel('Count')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Analyze Text Length\n",
    "train_df['text_length'] = train_df['text'].apply(lambda x: len(x.split()))\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.histplot(train_df['text_length'], kde=True, bins=30, color='purple')\n",
    "plt.title('Distribution of Message Lengths (Word Count)')\n",
    "plt.xlabel('Length')\n",
    "plt.show()\n",
    "\n",
    "print(f\"Max length: {train_df['text_length'].max()}\")\n",
    "print(f\"Average length: {train_df['text_length'].mean():.2f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Data Preprocessing\n",
    "1. **Tokenization**: Converting words to integers.\n",
    "2. **Padding**: Ensuring all sequences are the same length.\n",
    "3. **Encoding Labels**: Preparing target variables."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Parameters\n",
    "VOCAB_SIZE = 10000  # Max number of words to keep\n",
    "MAX_LEN = 60        # Max length of sequences (based on EDA ~50)\n",
    "OOV_TOKEN = \"<OOV>\" # Token for out-of-vocabulary words\n",
    "\n",
    "# Initialize Tokenizer with strict filters to handle contractions (e.g., I'm -> I m)\n",
    "tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN, filters='!\"#$%&()*+,-./:;<=>?@[\\\\]^_`{|}~\\t\\n\\'')\n",
    "tokenizer.fit_on_texts(train_df['text'])\n",
    "\n",
    "# Convert text to sequences\n",
    "train_sequences = tokenizer.texts_to_sequences(train_df['text'])\n",
    "val_sequences = tokenizer.texts_to_sequences(val_df['text'])\n",
    "test_sequences = tokenizer.texts_to_sequences(test_df['text'])\n",
    "\n",
    "# Pad sequences\n",
    "X_train = pad_sequences(train_sequences, maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "X_val = pad_sequences(val_sequences, maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "X_test = pad_sequences(test_sequences, maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "\n",
    "# Prepare Labels (Categorical)\n",
    "y_train = tf.keras.utils.to_categorical(train_df['label'], num_classes=6)\n",
    "y_val = tf.keras.utils.to_categorical(val_df['label'], num_classes=6)\n",
    "y_test = tf.keras.utils.to_categorical(test_df['label'], num_classes=6)\n",
    "\n",
    "print(\"Shape of X_train:\", X_train.shape)\n",
    "print(\"Shape of y_train:\", y_train.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. GloVe Embeddings\n",
    "Downloading and preparing pre-trained GloVe embeddings to provide semantic context."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Download GloVe embeddings (100d)\n",
    "if not os.path.exists('glove.6B.100d.txt'):\n",
    "    print('Downloading GloVe embeddings...')\n",
    "    url = 'http://nlp.stanford.edu/data/glove.6B.zip'\n",
    "    wget.download(url)\n",
    "    !unzip -q glove.6B.zip\n",
    "\n",
    "# Create Embedding Index\n",
    "embeddings_index = {}\n",
    "with open('glove.6B.100d.txt', encoding='utf-8') as f:\n",
    "    for line in f:\n",
    "        values = line.split()\n",
    "        word = values[0]\n",
    "        coefs = np.asarray(values[1:], dtype='float32')\n",
    "        embeddings_index[word] = coefs\n",
    "\n",
    "print(f'Found {len(embeddings_index)} word vectors.')\n",
    "\n",
    "# Create Embedding Matrix\n",
    "word_index = tokenizer.word_index\n",
    "embedding_matrix = np.zeros((VOCAB_SIZE, 100))\n",
    "for word, i in word_index.items():\n",
    "    if i < VOCAB_SIZE:\n",
    "        embedding_vector = embeddings_index.get(word)\n",
    "        if embedding_vector is not None:\n",
    "            embedding_matrix[i] = embedding_vector\n",
    "\n",
    "print(f'Embedding matrix shape: {embedding_matrix.shape}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Class Weights\n",
    "Handling imbalance by assigning higher weights to minority classes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "class_weights = compute_class_weight(\n",
    "    class_weight='balanced',\n",
    "    classes=np.unique(train_df['label']),\n",
    "    y=train_df['label']\n",
    ")\n",
    "class_weight_dict = dict(enumerate(class_weights))\n",
    "print(\"Class Weights:\", class_weight_dict)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Building the LSTM Model\n",
    "We use a **Bidirectional LSTM**. Bidirectional LSTMs train two LSTMs on the input sequence: one on the input sequence as-is, and another on a reversed copy. This provides more context.\n",
    "\n",
    "**Structure**:\n",
    "1. **Embedding**: Maps word indices to 100-dim vectors.\n",
    "2. **Bi-LSTM (64 units)**: Returns sequences for the next layer.\n",
    "3. **Bi-LSTM (32 units)**: Abstract features.\n",
    "4. **Dense & Dropout**: For classification and preventing overfitting."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "EMBEDDING_DIM = 100\n",
    "\n",
    "model = Sequential([\n",
    "    Input(shape=(MAX_LEN,)),\n",
    "    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, weights=[embedding_matrix], trainable=True),\n",
    "    Bidirectional(LSTM(64, return_sequences=True)),\n",
    "    BatchNormalization(),\n",
    "    Dropout(0.3),\n",
    "    Bidirectional(LSTM(32)),\n",
    "    BatchNormalization(),\n",
    "    Dropout(0.3),\n",
    "    Dense(64, activation='relu'),\n",
    "    Dropout(0.3),\n",
    "    Dense(6, activation='softmax')\n",
    "])\n",
    "\n",
    "model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])\n",
    "model.summary()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Model Training\n",
    "We use `EarlyStopping` to stop training when validation loss stops improving."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)\n",
    "reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)\n",
    "\n",
    "history = model.fit(\n",
    "    X_train, y_train,\n",
    "    validation_data=(X_val, y_val),\n",
    "    epochs=15,\n",
    "    batch_size=32,\n",
    "    callbacks=[early_stop, reduce_lr],\n",
    "    class_weight=class_weight_dict,\n",
    "    verbose=1\n",
    ")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Evaluation\n",
    "Visualizing the training performance and evaluating on the test set."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Plot Accuracy and Loss\n",
    "plt.figure(figsize=(12, 5))\n",
    "\n",
    "# Accuracy\n",
    "plt.subplot(1, 2, 1)\n",
    "plt.plot(history.history['accuracy'], label='Train Accuracy')\n",
    "plt.plot(history.history['val_accuracy'], label='Val Accuracy')\n",
    "plt.title('Model Accuracy')\n",
    "plt.legend()\n",
    "\n",
    "# Loss\n",
    "plt.subplot(1, 2, 2)\n",
    "plt.plot(history.history['loss'], label='Train Loss')\n",
    "plt.plot(history.history['val_loss'], label='Val Loss')\n",
    "plt.title('Model Loss')\n",
    "plt.legend()\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Evaluate on Test Data\n",
    "test_loss, test_acc = model.evaluate(X_test, y_test)\n",
    "print(f\"Test Accuracy: {test_acc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Classification Report\n",
    "y_pred_probs = model.predict(X_test)\n",
    "y_pred = np.argmax(y_pred_probs, axis=1)\n",
    "y_true = np.argmax(y_test, axis=1)\n",
    "\n",
    "print(classification_report(y_true, y_pred, target_names=label_map.values()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Confusion Matrix\n",
    "plt.figure(figsize=(8, 6))\n",
    "cm = confusion_matrix(y_true, y_pred)\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.values(), yticklabels=label_map.values())\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('True')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 11. Inference\n",
    "Test the model with custom input sentences."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def predict_emotion(text):\n",
    "    # Preprocess\n",
    "    seq = tokenizer.texts_to_sequences([text])\n",
    "    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')\n",
    "    \n",
    "    # Predict\n",
    "    pred_prob = model.predict(padded)\n",
    "    pred_label = np.argmax(pred_prob)\n",
    "    \n",
    "    return label_map[pred_label], pred_prob\n",
    "\n",
    "def debug_prediction(text):\n",
    "    seq = tokenizer.texts_to_sequences([text])\n",
    "    print(f\"Original: {text}\")\n",
    "    print(f\"Tokenized: {seq}\")\n",
    "    print(f\"Decoded: {tokenizer.sequences_to_texts(seq)}\")\n",
    "    return predict_emotion(text)\n",
    "\n",
    "# Test examples\n",
    "examples = [\n",
    "    \"I felt absolutely crushed when I heard the news.\",\n",
    "    \"I am so excited about the upcoming trip!\",\n",
    "    \"Why does this always happen to me? It's so frustrating!\",\n",
    "    \"I'm really afraid of what might happen next.\",\n",
    "    \"The party was a complete surprise!\",\n",
    "    \"I really love spending time with you.\"\n",
    "]\n",
    "\n",
    "for text in examples:\n",
    "    emotion, _ = predict_emotion(text)\n",
    "    print(f\"Text: {text} | Predicted Emotion: {emotion}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 12. Conclusion\n",
    "We successfully built a Bidirectional LSTM model for emotion detection.\\n\\n**Further Improvements:**\\n- Use **Pre-trained Embeddings** like GloVe or Word2Vec for better semantic understanding.\\n- Use **Transformer models** (BERT, RoBERTa) for state-of-the-art performance.\\n- **Data Augmentation** to handle class imbalance (though `dair-ai/emotion` is relatively balanced). "
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open('Emotion_Detection_From_Text_LSTM.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)
