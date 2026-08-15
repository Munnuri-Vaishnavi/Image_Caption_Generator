# 🖼️ Image Caption Generator

An end-to-end deep learning project that generates natural-language captions for images, built with a **ResNet50 (CNN) + LSTM (RNN)** architecture trained from scratch on the **Flickr8k** dataset — complete with a live, interactive web demo.

**🔗 Live demo:** [imagecaptiongenerator-flickr8k.streamlit.app](https://imagecaptiongenerator-flickr8k.streamlit.app)

---

## 📸 Screenshots

<!-- Add screenshots here, e.g.: -->
<!-- ![App screenshot](screenshots/demo.png) -->

*(Add a few screenshots of the app in action here — the upload tab, a generated caption, and the sample-image tab work well.)*

---

## ✨ Features

- **Upload your own image** or pick from a curated set of sample Flickr8k-style images
- **Beam search caption generation** (width 3) with length normalization — shows the top 3 candidate captions, not just one
- **Multi-language translation** of the generated caption (English, Telugu, Hindi, Tamil, Kannada)
- **Text-to-speech playback** of the caption in the selected language
- **Downloadable captioned image** — generates a PNG with the caption overlaid on the original image
- **Caption history** sidebar for quick reference during a session

---

## 🧠 How It Works

1. **Feature extraction** — a pretrained, frozen **ResNet50** (trained on ImageNet) converts each image into a 2048-dimensional feature vector, capturing its visual content.
2. **Caption generation** — an **LSTM (RNN)** takes that feature vector plus the words generated so far, and predicts the next word one step at a time, until it produces an end token.
3. **Beam search** — instead of greedily picking the single most likely next word, the model tracks multiple candidate sequences in parallel and returns the most probable overall caption.

```
Image → ResNet50 (CNN) → 2048-d feature vector → LSTM (RNN) → generated caption
```

### Known limitation

As a small-dataset, non-attention model, this architecture compresses the entire image into a single fixed-size vector before generating any text — so fine visual detail can be lost, even on images similar to the training data. This is the well-documented reason attention mechanisms and Vision Transformers were later developed for this task.

---

## 🗂️ Dataset

Trained on **[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)** — ~8,091 images, each with 5 human-written captions.

---

## 🏗️ Tech Stack

| Layer | Technology |
|---|---|
| Model training | TensorFlow / Keras (Google Colab, GPU) |
| Feature extractor | ResNet50 (pretrained on ImageNet) |
| Caption decoder | LSTM |
| Model hosting | [Hugging Face Hub](https://huggingface.co/vaishnavi22092006/Caption_.Generator) |
| Web app | Streamlit |
| Deployment | Streamlit Community Cloud |
| Translation | `deep-translator` (Google Translate) |
| Text-to-speech | `gTTS` |

---

## 📁 Repository Structure

```
.
├── app.py                          # Streamlit app (UI + inference)
├── requirements.txt                # Python dependencies
├── runtime.txt                     # Pinned Python version for deployment
├── caption_model_config.json       # Model hyperparameters (vocab size, max length, etc.)
├── Flickr8k_word_index.json        # Tokenizer word→index mapping
└── samples/                        # Sample images for the "Try a sample image" tab
```

> **Note:** The trained model file (`.keras`, ~103 MB) is hosted separately on [Hugging Face Hub](https://huggingface.co/vaishnavi22092006/Caption_.Generator) and downloaded automatically at runtime — it isn't stored in this repo.

---

## 🚀 Running Locally

```bash
git clone https://github.com/Munnuri-Vaishnavi/Image_Caption_Generator.git
cd Image_Caption_Generator
pip install -r requirements.txt
streamlit run app.py
```

The app will automatically download the trained model from Hugging Face Hub on first run.

---

## 🎓 Project Context

Built as part of a 12-week AI/ML Experiential Development Program (EDP), covering:
- **Weeks 4–7:** CNN feature extraction, RNN/LSTM sequence modeling, fine-tuning on Flickr8k
- **Week 8:** Deployment — packaging the trained model and building a demo-ready web app

---

## 🙏 Acknowledgments

- [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)
- [ResNet50 (Keras Applications)](https://keras.io/api/applications/resnet/)
- [Streamlit](https://streamlit.io/)
- [Hugging Face Hub](https://huggingface.co/)
