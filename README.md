## 🔊 DΞCIBΞLION

**OBINexus AI Psychoacoustic Classifier**
*A Constitutionally-Compliant Math-Driven Shouting Detector*

### 👁️‍🗨️ Overview

**DΞCIBΞLION** is an audio intelligence module forged in the labs of OBINexus, where noise meets logic and shouting is a feature, not a bug. It mathematically analyzes human vocal input to determine emotional projection through **log-scaled loudness evaluation**, using a sacred constant: `85 dB`.

Anything above that? It's shouting. Anything below? You're just passionate.

---

### 🧠 Features

* 📏 **Logarithmic Amplitude Scaling** – Faithful to psychoacoustic perception.
* 🔊 **Decibel-Based Emotion Segmentation** – 85 dB: The Great Divider.
* 🧮 **MFCC-Based CNN** – Because everything's better when it's filtered through 40 coefficients of raw judgment.
* 🧱 **Torch-Powered Architecture** – Low-latency yelling recognition for real-time emotional analytics.
* 🧰 **Dataset-ready Classifier** – Accepts raw `.wav` files like a champ.

---

### 📂 File Structure

* `ObinexusEmotionNet` – CNN architecture for binary classification (Not Shouting / Shouting)
* `VocalEmotionDataset` – MFCC + decibel tagging from audio files
* `apply_log_scale()` – Applies perceptual loudness transformation using `log₁₀(1 + α|x|)`
* `visualize_mfcc()` – See your screams visualized in all their MFCC glory

---

### 🚀 Usage

```bash
pip install numpy librosa torch matplotlib
```

```python
from decibelion import VocalEmotionDataset, ObinexusEmotionNet

# Load and visualize
visualize_mfcc('your_emotional_outburst.wav')

# Dataset and model
dataset = VocalEmotionDataset(['your_emotional_outburst.wav'])
model = ObinexusEmotionNet()

# Predict your shouting sins
x, y = dataset[0]
output = model(x.unsqueeze(0))  # Batch dimension
```

---

### 📡 The Law of the Loud

By OBINexus Constitutional Mandate:

* `dB >= 85` → **shouting**
* `dB < 85` → **tolerable emotional projection**
* `dB == 84` → **you, specifically**

---

### 📜 License

OBINexus ∞ MIT License with Embedded Shout Clause™
No yelling without constitutional justification.
