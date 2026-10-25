import io
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

st.set_page_config(page_title="Email Spam Detection", page_icon="📧", layout="wide")

st.title("📧 Email Spam Detection")
st.write(
    "Classify an email message as **SPAM** or **LEGITIMATE (HAM)** using "
    "TF-IDF features and a Multinomial Naive Bayes model."
)

st.info(
    "This project dataset is synthetic and intended for demonstration. "
    "Model scores do not represent guaranteed real-world performance."
)

def load_dataset(uploaded_file):
    """Load CSV or Excel data and validate the required text/label columns."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()
    if name.endswith(".csv"):
        data = pd.read_csv(io.BytesIO(raw))
    elif name.endswith((".xlsx", ".xls")):
        data = pd.read_excel(io.BytesIO(raw))
    else:
        raise ValueError("Please upload a CSV or Excel (.xlsx/.xls) file.")

    # Normalize column names to make the app tolerant of capitalization/spaces.
    data.columns = [str(col).strip().lower() for col in data.columns]
    if not {"text", "label"}.issubset(data.columns):
        raise ValueError("Dataset must contain columns named 'text' and 'label'.")

    data = data[["text", "label"]].copy()
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data["label"] = data["label"].fillna("").astype(str).str.strip().str.lower()
    original_rows = len(data)
    data = data[(data["text"] != "") & data["label"].isin(["spam", "ham"])]
    removed_rows = original_rows - len(data)
    duplicate_rows = int(data.duplicated(subset=["text", "label"]).sum())
    data = data.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

    if len(data) < 10:
        raise ValueError("At least 10 valid rows are required to train the model.")
    if data["label"].nunique() != 2:
        raise ValueError("The dataset must contain both labels: 'spam' and 'ham'.")
    if data["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified split.")
    return data, removed_rows, duplicate_rows

@st.cache_resource
def train_model(texts, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        list(texts),
        list(labels),
        test_size=0.20,
        random_state=42,
        stratify=list(labels),
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True, stop_words="english", ngram_range=(1, 2)
        )),
        ("classifier", MultinomialNB()),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, y_test, predictions, len(X_train), len(X_test)

st.sidebar.header("1. Load training data")
uploaded = st.sidebar.file_uploader(
    "Upload your email dataset (CSV or Excel)",
    type=["csv", "xlsx", "xls"],
    help="Required columns: text and label. Labels must be spam or ham.",
)

default_path = Path(__file__).with_name("email_spam_dataset_500.xlsx")
data = None
removed_rows = duplicate_rows = 0

try:
    if uploaded is not None:
        data, removed_rows, duplicate_rows = load_dataset(uploaded)
    elif default_path.exists():
        data, removed_rows, duplicate_rows = load_dataset(
            type("LocalUpload", (), {
                "name": default_path.name,
                "getvalue": lambda self: default_path.read_bytes()
            })()
        )
        st.sidebar.success("Using email_spam_dataset_500.xlsx from the app folder.")
    else:
        st.warning(
            "Upload your dataset in the sidebar to begin. "
            "You can also place email_spam_dataset_500.xlsx beside app.py."
        )
        st.stop()
except Exception as exc:
    st.error(f"Could not load dataset: {exc}")
    st.stop()

st.sidebar.caption(f"Valid examples: {len(data)}")
st.sidebar.caption(f"Invalid/empty rows removed: {removed_rows}")
st.sidebar.caption(f"Duplicate rows removed: {duplicate_rows}")

try:
    model, y_test, y_pred, train_count, test_count = train_model(
        tuple(data["text"]), tuple(data["label"])
    )
except Exception as exc:
    st.error(f"Could not train the model: {exc}")
    st.stop()

tab_predict, tab_metrics, tab_data = st.tabs(
    ["🔎 Check an Email", "📊 Model Evaluation", "🗂️ Dataset"]
)

with tab_predict:
    st.subheader("Check an email message")
    email_text = st.text_area(
        "Paste the email text here",
        value="Congratulations! You won a free gift card. Claim now!",
        height=180,
        placeholder="Paste the subject and body of the email...",
    )
    if st.button("Detect Spam", type="primary", use_container_width=False):
        if not email_text.strip():
            st.warning("Please enter an email message first.")
        else:
            prediction = model.predict([email_text.strip()])[0]
            probabilities = model.predict_proba([email_text.strip()])[0]
            class_names = model.named_steps["classifier"].classes_
            probability_map = dict(zip(class_names, probabilities))
            spam_score = float(probability_map.get("spam", 0.0))
            if prediction == "spam":
                st.error("🚨 Prediction: SPAM")
            else:
                st.success("✅ Prediction: LEGITIMATE (HAM)")
            st.metric("Model score for spam", f"{spam_score:.1%}")
            st.progress(min(max(spam_score, 0.0), 1.0))
            st.caption(
                "The score is the model's estimated probability, not a guarantee. "
                "Do not open suspicious links or attachments based only on this result."
            )

with tab_metrics:
    st.subheader("Held-out test-set evaluation")
    st.caption(
        f"Training examples: {train_count} · Test examples: {test_count} · "
        "80/20 stratified split with random_state=42"
    )
    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.1%}")
    report = classification_report(
        y_test, y_pred, labels=["ham", "spam"], output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report).T
    st.write("Precision, recall and F1-score")
    st.dataframe(report_df.round(3), use_container_width=True)

    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual ham", "Actual spam"],
        columns=["Predicted ham", "Predicted spam"],
    )
    st.write("Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    false_positives = sum(
        actual == "ham" and predicted == "spam"
        for actual, predicted in zip(y_test, y_pred)
    )
    st.metric("False positives", int(false_positives))
    st.caption("False positives are legitimate emails incorrectly marked as spam.")

with tab_data:
    st.subheader("Training dataset overview")
    left, right = st.columns(2)
    with left:
        st.metric("Valid rows", len(data))
        st.metric("Spam examples", int((data["label"] == "spam").sum()))
    with right:
        st.metric("Legitimate examples", int((data["label"] == "ham").sum()))
        st.metric("Unique messages", int(data["text"].nunique()))
    st.write("Label distribution")
    st.bar_chart(data["label"].value_counts())
    with st.expander("Preview dataset"):
        st.dataframe(data.head(50), use_container_width=True)

st.divider()
st.caption(
    "Method: TF-IDF vectorization (unigrams and bigrams) + Multinomial Naive Bayes. "
    "This is an educational prototype; use a representative, appropriately licensed "
    "real-world dataset before drawing conclusions about real email safety."
)
