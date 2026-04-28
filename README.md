# Sentiment Analysis API

A lightweight sentiment analysis API built with **FastAPI** and **TextBlob**. This project demonstrates core AI engineering principles including robust input validation, threshold-based classification, reproducible testing, and structured error analysis.

---

## Tech Stack

| Component   | Technology |
| ----------- | ---------- |
| Framework   | FastAPI    |
| NLP Library | TextBlob   |
| Validation  | Pydantic   |
| Server      | Uvicorn    |

---

## Setup and Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/sentiment-analysis-api.git
cd sentiment-analysis-api
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

> **Windows users:** use `venv\Scripts\activate`

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLP Corpora

TextBlob requires specific NLTK data packages to function. Run this command **once**:

```bash
python -m textblob.download_corpora
```

---

## Usage

### Option 1: Run Local Tests

Execute the test suite directly in the terminal to view predictions for 12 curated sentences.

```bash
python main.py
```

### Option 2: Start the API Server

Run the FastAPI server with auto-reload enabled.

```bash
uvicorn main:app --reload
```

Once running, access the interactive API documentation at:

```
http://127.0.0.1:8000/docs
```

---

## Test Results

The following table shows the results of the 12 test cases using a polarity threshold of **+/- 0.1**.

| Input Text                                            | Polarity | Predicted Sentiment |
| ----------------------------------------------------- | -------- | ------------------- |
| I absolutely love this new feature!                   | 0.3352   | Positive            |
| The customer service was incredibly helpful and fast. | 0.5500   | Positive            |
| Had a wonderful experience, highly recommend!         | 0.6000   | Positive            |
| This product exceeded all my expectations.            | 0.0000   | Neutral             |
| The app crashes every time I open it.                 | 0.0000   | Neutral             |
| Terrible quality, completely wasted my money.         | -0.6000  | Negative            |
| I'm very disappointed with the delivery delay.        | -0.9750  | Negative            |
| The staff was rude and unprofessional.                | -0.3000  | Negative            |
| The package arrived on Tuesday.                       | 0.0000   | Neutral             |
| I ordered the blue shirt, size medium.                | 0.0000   | Neutral             |
| The meeting is scheduled for 3 PM tomorrow.           | 0.0000   | Neutral             |
| The book has 320 pages and was published in 2022.     | 0.0000   | Neutral             |

### Threshold Logic

```
Polarity >  0.1  ->  Positive
Polarity < -0.1  ->  Negative
Otherwise        ->  Neutral
```

---

## Error and Uncertainty Analysis

### False Neutral (Positive Context)

**Input:** `"This product exceeded all my expectations."`  
**Result:** Neutral (0.0000)

**Analysis:** TextBlob's lexicon-based approach failed to capture the strong positive implication of "exceeded expectations." The model likely treated the sentence as factual due to the absence of explicit emotional adjectives such as "good" or "great" in its immediate context. A transformer-based model (e.g., BERT or RoBERTa) would likely handle this contextual nuance better through deeper semantic understanding.

---

### False Neutral (Negative Context)

**Input:** `"The app crashes every time I open it."`  
**Result:** Neutral (0.0000)

**Analysis:** The model missed the negative sentiment associated with "crashes." In a technical context, "crash" implies failure and user frustration, but a simple lexicon model may not assign it a strong negative weight compared to emotionally charged words like "terrible" or "hate." This highlights a core limitation of bag-of-words approaches in domain-specific sentiment tasks.

---

## Project Structure

```
sentiment-analysis-api/
├── main.py              # FastAPI application, input validation, sentiment logic, and test runner
├── requirements.txt     # Python dependencies
└── .gitignore           # Standard Python ignore rules
```

---

## Demo Video

[Link to demo video here]

---

## License

This project is open source and available under the [MIT License](LICENSE).
