# Urdu → Roman Urdu Neural Machine Translation

## 📌 Project Overview
This project implements a **Neural Machine Translation (NMT)** model to transliterate **Urdu → Roman Urdu** using a **BiLSTM encoder–decoder** architecture. The task was defined in the course assignment【5†NLP_Assignment1_Urdu_to_RomanUrdu.pdf】.

Evaluation metrics include:
- **BLEU Score**
- **Perplexity**
- **Character Error Rate (CER)**
- **Edit Distance**

---

## 📂 Dataset & Preprocessing
- **Dataset**: [`urdu_ghazals_rekhta`](https://github.com/amir9ume/urdu_ghazals_rekhta)
- **Steps performed**:
  - Urdu text cleaning and normalization.
  - Vocabulary optimized to **512 tokens** using **Byte Pair Encoding (BPE)** → improved generalization and reduced OOV.
  - Train/Validation/Test split: **50% / 25% / 25%**.

---

## ⚙️ Experimentation
We experimented with:
- **Embedding sizes**: 128–512
- **Hidden dimensions**: 128–256
- **Dropout rates**: 0.1–0.3
- **Encoder layers**: 2
- **Decoder layers**: 3–4

🔑 Key approaches:
- Feeding **source tokens into both encoder and decoder** (major boost).
- Training decoder **more heavily than encoder** (sometimes freezing encoder).
- Using deeper **decoder than encoder**.

---

## 📊 Results

| Model | Encoder–Decoder Setup | Loss | Perplexity | BLEU | CER | Edit Dist. | Key Notes |
|-------|----------------------|------|------------|------|-----|------------|-----------|
| **M1 (Baseline)** | BiLSTM Encoder (2L, 128) → LSTM Decoder (4L, hidden=256) | **3.0586** | **21.2971** | **0.2745** | **0.3691** | – | First working model; intuition of feeding `src` into decoder started here. |
| **M2 (Best)** | BiLSTM Encoder (2L, 128) → LSTM Decoder (4L, hidden=256) with projection & src fed to decoder | **2.0089** | **7.4549** | **0.4518** | **0.1949** | **10.96** | Freezing encoder, deeper decoder. Best generalization & performance. |
| **M3** | BiLSTM Encoder (2L, 128) → LSTM Decoder (4L, hidden=256) with concat states | **2.0439** | **7.7209** | **0.3836** | **0.2742** | **14.85** | Stable training but lower BLEU vs M2. |
| **M4** | BiLSTM Encoder (2L, 256) → LSTM Decoder (3L, hidden=256) | ~**2.04** | ~**7.71** | **0.44** | **0.22** | 11.84 | Larger embedding, but weaker decoder depth → worse results. |

---

## 📌 Key Findings
1. **512-token BPE vocabulary** improved performance significantly.
2. Feeding **source into decoder** was crucial for better alignments.
3. **Decoder depth > Encoder depth** gave better transliteration accuracy.
4. **Freezing encoder weights** sometimes stabilized training.
5. **Model M2 consistently outperformed all others** across BLEU, CER, and perplexity.

---

## 🏆 Final Model (M2)
- **BiLSTM Encoder** (2 layers, hidden=128)
- **LSTM Decoder** (4 layers, hidden=256, dropout=0.3)
- **Evaluation Metrics**:
  - **Test Loss** = **2.01**
  - **Perplexity** = **7.45**
  - **BLEU Score** = **0.45**
  - **CER** = **0.19**
  - **Avg. Edit Distance** = **10.96**

✅ This architecture balanced efficiency and performance.

---

## 🚀 Deployment
A **Streamlit application** was built around **M2**:
- Input: Urdu text
- Output: Roman Urdu transliteration
- Side-by-side comparison with ground truth (for demo/testing)
- Lightweight UI for deployment

---

## ✅ Conclusion
Through multiple experiments, we concluded that:
- **Smaller encoder + deeper decoder with source-fed decoding** works best for Urdu → Roman Urdu transliteration.
- The final **M2 model** achieved **BLEU = 0.45** and **CER = 0.19**, proving effective for low-resource poetic text.
- Deployed via **Streamlit** for real-time usage.


---
