# SupportIQ — Customer Support Ticket Classification & Priority Prediction

SupportIQ is a machine-learning web application that automatically classifies customer support tickets into categories and predicts their priority as High, Medium, or Low. It also stores submitted tickets and their predictions for later review.

## Features

- Predicts one of 10 customer support categories.
- Estimates ticket priority as High, Medium, or Low.
- Provides a web dashboard for submitting support tickets.
- Stores ticket predictions in an SQLite database.
- Displays recent ticket history and reference numbers.
- Provides an API for ticket predictions.
- Includes model evaluation reports and confusion matrices.

## Machine Learning Approach

The project uses two separately trained Linear Support Vector Machine (Linear SVM) classifiers:

1. **Category classifier:** Predicts the support department or ticket category.
2. **Priority classifier:** Predicts the ticket priority.

Ticket text is converted into numerical features using TF-IDF (Term Frequency–Inverse Document Frequency). Unigrams and bigrams are used to represent individual words and two-word combinations.

The dataset was cleaned to retain English-language tickets and prepare the text and target labels for training.

## Dataset

- Source: Hugging Face — `vasu1111/customer-support-tickets`
- English tickets used: 16,338
- Categories: 10
- Priority levels: 3

The dataset contains customer support ticket text and associated labels. Refer to the original dataset page for its licensing and usage terms.

## Evaluation Results

The selected Linear SVM models were evaluated on held-out test data.

| Metric | Category Classifier | Priority Classifier |
|---|---:|---:|
| Test samples | 3,268 | 3,268 |
| Accuracy | 68.27% | 71.21% |
| Macro F1-score | 69.82% | 70.40% |
| Weighted F1-score | 68.26% | 71.15% |

The evaluation also includes per-class precision, recall, F1-scores, and confusion matrices.

These results are measured on held-out data. They do not guarantee the same performance on new real-world tickets.

## Technology Stack

- **Language:** Python
- **Data processing:** pandas
- **Machine learning:** scikit-learn
- **Text representation:** TF-IDF
- **Models:** Linear SVM
- **Backend API:** FastAPI
- **Database:** SQLite
- **Frontend:** HTML, CSS, JavaScript

## Project Structure

```text
Support_Ticket_Intelligence/
├── data/
│   ├── tickets.csv
│   ├── support_tickets_raw.csv
│   └── tickets_clean.csv
├── models/
│   ├── ticket_classifier.joblib
│   ├── priority_classifier.joblib
│   ├── category_model_comparison.csv
│   ├── priority_model_comparison.csv
│   └── evaluation/
├── static/
│   └── index.html
├── database.py
├── main.py
├── prepare_data.py
├── train_models.py
├── evaluate_models.py
├── requirements.txt
└── README.md
```

## Setup and Run

### 1. Clone the repository

```bash
git clone YOUR_REPOSITORY_URL
cd Support_Ticket_Intelligence
```

Replace `YOUR_REPOSITORY_URL` with your actual repository URL.

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Start the application

```powershell
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

The API documentation is available at `http://127.0.0.1:8000/docs`.

## Model Evaluation

To generate evaluation reports and confusion matrices:

```powershell
python evaluate_models.py
```

Evaluation outputs are saved in `models/evaluation/`.

To retrain the models, review `train_models.py` and run the training command documented in that script. Retraining may replace the saved model files.

## Limitations

- Predictions may be incorrect, especially for categories with similar language.
- Priority predictions are estimates and should not replace human review for urgent or sensitive customer issues.
- Performance depends on the quality and representativeness of the training dataset.
- The models were evaluated on held-out data from the same dataset; performance on live business data may differ.

## Future Improvements

- Improve classification of similar ticket categories.
- Evaluate additional text-classification models.
- Test performance on a separate, real-world dataset.
- Add confidence-based review for uncertain predictions.
- Explore model monitoring and retraining as new labelled tickets become available.

## Author

Developed as a machine-learning project to explore text classification, priority prediction, model evaluation, and API-based deployment.
