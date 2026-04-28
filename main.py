import logging
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from textblob import TextBlob

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sentiment Analysis Mini Project")


class TextInput(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("Input must be a string.")
        if not v.strip():
            raise ValueError("Input cannot be empty or whitespace-only.")
        return v.strip()


def predict_sentiment(text: str) -> dict:
    """Returns sentiment label based on TextBlob polarity thresholds."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {"text": text, "polarity": round(polarity, 4), "sentiment": label}


@app.post("/predict")
def get_sentiment(input_data: TextInput):
    try:
        return predict_sentiment(input_data.text)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail="Internal prediction error")


TEST_CASES = [
    # Positive
    "I absolutely love this new feature!",
    "The customer service was incredibly helpful and fast.",
    "Had a wonderful experience, highly recommend!",
    "This product exceeded all my expectations.",
    # Negative
    "The app crashes every time I open it.",
    "Terrible quality, completely wasted my money.",
    "I'm very disappointed with the delivery delay.",
    "The staff was rude and unprofessional.",
    # Neutral
    "The package arrived on Tuesday.",
    "I ordered the blue shirt, size medium.",
    "The meeting is scheduled for 3 PM tomorrow.",
    "The book has 320 pages and was published in 2022.",
]


def run_tests() -> list[dict]:
    """Executes test suite and returns structured results."""
    logger.info("Running sentiment analysis on 12 test cases...")
    return [predict_sentiment(sentence) for sentence in TEST_CASES]


def format_results(results: list[dict]) -> str:
    """Formats results into a fixed-width table for CLI output."""
    header = f"{'SENTIMENT':<10} | {'POLARITY':>8} | INPUT TEXT"
    separator = "-" * 85
    rows = [header, separator]
    for res in results:
        rows.append(f"{res['sentiment']:<10} | {res['polarity']:>8.4f} | {res['text']}")
    rows.append(separator)
    return "\n".join(rows)


if __name__ == "__main__":
    test_results = run_tests()
    formatted_output = format_results(test_results)
    print(formatted_output)
    logger.info("Test execution complete.")