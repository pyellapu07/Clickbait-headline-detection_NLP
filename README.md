# Clickbait Headline Detection with DistilBERT  
*(ACL-style NLP Course Project)*

## Overview
This repository contains code and artifacts for an academic Natural Language Processing project on **clickbait headline detection**. We formulate the task as a **binary text classification problem** (clickbait vs. legitimate) and fine-tune a **DistilBERT** model to capture both stylistic and semantic signals present in news headlines.

The project focuses on:
- transformer-based modeling for short text classification,
- rigorous evaluation using standard NLP metrics,
- qualitative error analysis and visualization, and
- a lightweight **Streamlit application** for interactive inference.

---

## Authors
- **Pradeep Yellapu**  
- **Harshitha Murali**  
- **Sriniketh Shankar**  
- **Girik**

**Affiliation:** University of Maryland

---

## Repository Structure
```text
.
├── app.py                 # Streamlit app for interactive headline classification
├── figures/               # Visualizations used in the ACL-style report
│   ├── label_distribution.png
│   ├── character_length.png
│   ├── exclamation_marks_by_class.png
│   ├── question_marks_by_class.png
│   └── confusion_matrix.png
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore
```
Note:
Large artifacts such as trained model checkpoints, datasets, and PDFs are intentionally excluded from the repository to comply with GitHub file size limits.

## Model Info
- Base Model: DistilBERT (distilbert-base-uncased)
- Task: Binary classification (Clickbait vs. Legitimate)
- Input: News headlines
- Output: Predicted class label with confidence score
- Training: Fine-tuned on a labeled headline dataset
- Evaluation Metrics: Accuracy, Precision, Recall, F1-score, Confusion Matrix

The model captures linguistic cues such as sensational phrasing, punctuation patterns, and length-based signals while retaining contextual understanding through transformer representations.

## Running the Streamlit Demo
1. Environment Setup

We recommend using a virtual environment.
```
python -m venv venv
source venv/bin/activate        # macOS/Linux
# OR
venv\Scripts\activate           # Windows
```

2. Install Dependencies
```
pip install -r requirements.txt
```
3. Model Checkpoint (Required)
The Streamlit app expects a locally available fine-tuned DistilBERT checkpoint.

Download the model and supporting files from Google Drive:
[[Drive link](https://drive.google.com/drive/folders/1Ns9MemiFCuGVZtnI0ENhFdoKsiM6ttnC?usp=drive_link)]

Expected directory structure:
```
distilbert_clickbait/
└── checkpoint-800/
    ├── config.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── vocab.txt
```

Place the distilbert_clickbait/ directory in the project root before running the app.

4. Launch the App
streamlit run app.py

The application will open in your browser and allow you to input headlines for real-time clickbait detection.

## Reproducibility Notes

- This repository is intended for evaluation and demonstration, not full retraining.
- All reported results are computed on a held-out test set.
- Tokenization, preprocessing, and random seeds are fixed and described in the accompanying report.

## Responsible NLP Considerations

This project is intended strictly for academic and educational purposes.

We explicitly consider:
- ambiguity in clickbait labeling,
- stylistic and topical bias in headlines,
- overconfidence of neural classifiers, and limitations of automated content moderation systems.

The Streamlit demo should not be interpreted as a production-ready moderation tool.

## Citation
If you reference this work in an academic context, please cite it as a course project:

Yellapu, P., Murali, H., Shankar, S., & Girik. (2025).
Clickbait Headline Detection with DistilBERT.
University of Maryland, NLP Course Project.

## Acknowledgments
This project was completed as part of a graduate-level Natural Language Processing course at the University of Maryland.
