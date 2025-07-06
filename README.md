# Emotion Recognition using Convolutional Neural Networks (CNNs)

This project represents an academic research into deep learning-based facial emotion recognition, carried out as part of the coursework for the Artificial Neural Networks (ANN) course in university. It focuses on facial emotion recognition using Convolutional Neural Networks (CNNs), along with a comparative analysis of five state-of-the-art deep learning models.


---

##  Objective

To develop a facial emotion recognition system capable of identifying emotions such as anger, happiness, sadness, surprise, and neutrality from grayscale facial images using CNNs.

---

## Dataset

- **Name:** Face Expression Recognition Dataset  
- **Source:** [Kaggle Link](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)  
- **Image Size:** 48x48 pixels, grayscale  
- **Emotions Covered:** Anger, Disgust, Fear, Happy, Sad, Surprise, Neutral  

---

## 🛠Preprocessing

1. **Grayscale Conversion** – Images are preloaded as grayscale (48x48).  
2. **Normalization** – Pixel values scaled to [0, 1].  
3. **Label Encoding** – Emotions encoded into integers.  
4. **One-Hot Encoding** – For multi-class classification.  

---

## Proposed CNN Architecture

- **Conv2D Layer 1:** 128 filters, 3x3 kernel, ReLU  
- **Conv2D Layers (Deeper):** 256, 512, 512 filters, ReLU  
- **MaxPooling2D:** Pool size 2x2  
- **Dropout:** 0.3 (dense), 0.4 (conv)  
- **Flatten Layer**  
- **Dense Output Layer** with softmax  

**Loss Function:** Categorical Crossentropy  
**Optimizer:** Adam  
**Metrics:** Accuracy, Precision, Recall, F1-score  

---

## Comparative Study

We compared our custom CNN with five advanced models on the same dataset:

| Model         | Accuracy (val/test) | Notes                                      |
|---------------|---------------------|--------------------------------------------|
| DenseNet      | ~99.95%       | High train accuracy, potential overfitting |
| HighwayNet    | ~62.5%        | Moderate accuracy, better generalization   |
| Wide ResNet   | ~25.29%       | Underfitting or poor generalization        |
| Pyramidal Net | 89.32%        | Moderate performance               |
| VGG16         | ~25.9%        | Performance needs tuning                   |

📚 **Reference Papers:**
- [Base Paper](https://iopscience.iop.org/article/10.1088/1742-6596/1962/1/012040/pdf)
- [Paper 1](https://iopscience.iop.org/article/10.1088/1742-6596/2236/1/012004/pdf)
- [Paper 2](https://iopscience.iop.org/article/10.1088/1361-6501/ad191c)
- [Paper 3](https://iopscience.iop.org/article/10.1088/2057-1976/ac107c)
- [Paper 4](https://iopscience.iop.org/article/10.1088/1741-2552/ac49a7)
- [Paper 5](https://link.springer.com/article/10.1007/s11042-023-14753-y)

---

## 🧪 Evaluation

- **Confusion Matrix**
- **Classification Report**
- **Training Curves (Accuracy/Loss over Epochs)**
- **Model Comparison Charts**


---

