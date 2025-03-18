# GAAN - Music Genre Classification

## 📌 Project Overview
GAAN is a deep learning-based music genre classification system. This project utilizes Convolutional Neural Networks (CNNs) to classify music tracks into different genres based on Mel-Frequency Cepstral Coefficients (MFCC) extracted from audio files.

## 🎵 Dataset
- **Dataset Used:** [GTZAN Music Genre Dataset](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification)
- The dataset contains **1,000** music clips of **30 seconds** each, categorized into **10 genres**.
- Features are extracted using **MFCCs** to train the model.

## ⚙️ Installation & Setup
### Prerequisites
Ensure you have Python installed along with the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Project
1. **Preprocess the Dataset** (Extract MFCC features)
   ```bash
   python src/datapreprocess.py
   ```
2. **Train the Model**
   ```bash
   python src/main.py
   ```
3. **Make Predictions**
   ```bash
   python src/main.py --predict
   ```

## 🧠 Model Architecture
The CNN model consists of:
- **3 Convolutional Layers** with ReLU activation and MaxPooling
- **Batch Normalization** for stable training
- **Fully Connected Layers** leading to a Softmax output (10 genres)

## 🚀 Future Improvements
- Fine-tune the model with additional audio features.
- Optimize for real-time music classification.
- Deploy as a web app or mobile application.

## 🤝 Contributors
- **Alvi Anowar** ([GitHub](https://github.com/alvianowar))
- **Md.T** ([GitHub](https://github.com/MdTausifulBari))

---
📌 *This project was developed as part of CSE-499 Senior Design.*

