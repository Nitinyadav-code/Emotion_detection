# Emotion Detection from Text using LSTM - Project Guide

This project is a Deep Learning application that detects emotions (Anger, Fear, Joy, Love, Sadness, Surprise) from text messages using a **Bidirectional LSTM** neural network. This guide explains every part of the code in simple terms.

---

## 🏗️ 1. Project Overview

We use **Natural Language Processing (NLP)** to teach a computer to understand human emotions.
*   **Input**: A sentence (e.g., "I am feeling great!")
*   **Model**: A Long Short-Term Memory (LSTM) network that remembers context.
*   **Output**: The predicted emotion (e.g., "Joy").

---

## 📖 2. Code Breakdown (Cell by Cell)

### **Step 1: Setup and Imports**
We import the tools we need. think of these as the "ingredients" for our recipe.

*   `numpy`, `pandas`: For handling data tables and numbers.
*   `matplotlib`, `seaborn`: For drawing graphs to visualize data.
*   `tensorflow`: The main brain (Deep Learning library) to build the model.
*   `datasets`: To download the emotion dataset from Hugging Face.
*   `Tokenizer`: Converts text into numbers (computers understand numbers, not words).
*   `pad_sequences`: Makes sure all sentences are the same length (required for the model).

### **Step 2: Data Loading**
We load the **`dair-ai/emotion`** dataset. It comes split into three parts:
1.  **Train**: To teach the model.
2.  **Validation**: To test the model *during* training (to tune it).
3.  **Test**: To test the model *after* training (final exam).

### **Step 3: Exploratory Data Analysis (EDA)**
Before training, we look at the data.
*   **Class Distribution**: We check if we have enough examples for each emotion. If we have too much "Joy" and little "Surprise", the model might become biased.
*   **Text Length**: We check how long the sentences are to decide how much to "pad" them.

### **Step 4: Data Preprocessing**
*   **`Tokenizer`**: We create a dictionary of the 10,000 most common words.
    *   *Example*: "I love AI" -> `[1, 5, 203]`
    *   *Fix*: We added specific **filters** to handle contractions (like "I'm" or "don't") so they aren't lost.
*   **`pad_sequences`**: Using `maxlen=60`.
    *   If a sentence is 10 words, we add 50 zeros (padding).
    *   If a sentence is 100 words, we cut off the last 40 (truncating).

### **Step 5: GloVe Embeddings (The "Brain" Upgrade)**
Instead of learning English from scratch, we use **GloVe**.
*   **What is it?**: A pre-trained list of 100 numbers for every English word, learned from reading Wikipedia.
*   **Why use it?**: It knows that "happy" and "joy" are similar because their numbers are close. This drastically improves accuracy.

### **Step 6: Class Weights**
*   **Problem**: Some emotions appear less often.
*   **Solution**: We calculate **weights**. We tell the model: "If you get a rare emotion wrong, you get a BIG penalty. If you get a common one wrong, a small penalty." This forces it to learn everything equally.

---

## 🧠 3. The Model Architecture (The "Brain")

We build the model layer by layer using `Sequential`:

1.  **Input Layer**: Accepts the list of 60 numbers (our sentence).
2.  **Embedding Layer**: Swaps the word numbers for the rich GloVe vectors.
3.  **Bidirectional LSTM**:
    *   **LSTM (Long Short-Term Memory)**: A type of brain cell that remembers past words to understand the full sentence structure.
    *   **Bidirectional**: It reads the sentence **Forward** (Start -> End) AND **Backward** (End -> Start).
    *   *Why?* To understand "not happy", you need to see both "not" (past) and "happy" (future).
4.  **BatchNormalization**: Keeps the numbers stable so training is faster.
5.  **Dropout**: Randomly turns off 30% of neurons during training to stop the model from memorizing answers (Overfitting).
6.  **Dense Output Layer**: The final decision maker. It has 6 neurons (one for each emotion) using **Softmax**.
    *   **Softmax**: Converts numbers into probabilities (e.g., Joy: 80%, Sadness: 10%...).

---

## ⚙️ 4. Training Concepts Explained

### **Loss Function: `categorical_crossentropy`**
This measures **how wrong** the model is.
*   If the true emotion is "Joy" and model predicts "Sadness", Loss is **HIGH**.
*   The goal of training is to make Loss **LOW**.

### **Optimizer: `adam`**
This is the "teacher". It looks at the Loss and adjusts the model's brain connections (weights) to improve the next guess.

### **Callbacks (Smart Training Tools)**
1.  **`EarlyStopping`**:
    *   *Stop Condition*: "If the model stops improving for 3 epochs (rounds), STOP training."
    *   *Why?* prevents wasting time and overfitting (getting worse by trying too hard).
2.  **`ReduceLROnPlateau`**:
    *   *Action*: "If the model gets stuck, slow down the learning rate."
    *   *Analogy*: It's like trying to park a car. When you get close, you drive slower to park perfectly.

---

## 🔮 5. Inference
This is where we use the model!
*   **`predict_emotion`**: Takes raw text -> Tokenizes it -> Pads it -> Feeds to Model -> Returns the Emotion with the highest percentage.
*   **`debug_prediction`**: Shows exactly what the computer sees (the list of numbers) to help us fix issues if it predicts wrong.

---

### **Summary**
1.  **Load Data** (Tweets).
2.  **Clean Data** (Convert to numbers).
3.  **Add Knowledge** (GloVe Vectors).
4.  **Build Brain** (Bi-LSTM).
5.  **Train** (with smart stopping).
6.  **Predict** (Classify new text).
