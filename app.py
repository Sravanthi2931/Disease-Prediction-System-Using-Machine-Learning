from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Healthcare.csv"

app = Flask(__name__, template_folder=str(BASE_DIR / "Template"))

GENDER_MAP = {"male": 1, "female": 0, "other": 2}


def load_healthcare_data(path: Path) -> pd.DataFrame:
    """
    The dataset ships with a .csv extension but is actually an Excel workbook.
    Try parsing as CSV first and gracefully fall back to read_excel.
    """
    try:
        return pd.read_csv(path)
    except (UnicodeDecodeError, pd.errors.ParserError):
        return pd.read_excel(path)


def tokenize_symptoms(symptom_text: str | float | None) -> List[str]:
    if symptom_text is None:
        return []
    return [
        token.strip().lower()
        for token in str(symptom_text).split(",")
        if token.strip()
    ]


def build_training_frame() -> tuple[
    pd.DataFrame, pd.Series, MultiLabelBinarizer, List[str]
]:
    raw_df = load_healthcare_data(DATA_PATH)
    required_columns = {"Age", "Gender", "Symptoms", "Symptom_Count", "Disease"}
    missing = required_columns.difference(raw_df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")

    df = raw_df.dropna(subset=required_columns).copy()
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Symptom_Count"] = pd.to_numeric(df["Symptom_Count"], errors="coerce")
    df = df.dropna(subset=["Age", "Symptom_Count"])

    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower()
    df["Gender_Code"] = (
        df["Gender"].map(GENDER_MAP).fillna(GENDER_MAP["other"]).astype(int)
    )

    symptom_lists = df["Symptoms"].apply(tokenize_symptoms)
    binarizer = MultiLabelBinarizer()
    symptom_matrix = binarizer.fit_transform(symptom_lists)
    symptom_columns = [f"sym_{token}" for token in binarizer.classes_]

    symptom_df = pd.DataFrame(symptom_matrix, columns=symptom_columns, index=df.index)
    feature_df = pd.concat(
        [df[["Age", "Gender_Code", "Symptom_Count"]], symptom_df], axis=1
    )
    labels = df["Disease"].astype(str)
    feature_df.columns = feature_df.columns.astype(str)

    return feature_df, labels, binarizer, symptom_columns


def assemble_features(
    age: int,
    gender: str,
    symptom_count: int,
    symptoms: Iterable[str],
) -> pd.DataFrame:
    gender_code = GENDER_MAP.get(gender.lower(), GENDER_MAP["other"])
    numeric_df = pd.DataFrame(
        [[age, gender_code, symptom_count]],
        columns=["Age", "Gender_Code", "Symptom_Count"],
    )

    symptom_vector = SYMPTOM_BINARIZER.transform([symptoms])
    symptom_df = pd.DataFrame(symptom_vector, columns=SYMPTOM_COLUMNS)

    features = pd.concat([numeric_df, symptom_df], axis=1)
    missing_cols = [col for col in FEATURE_COLUMNS if col not in features.columns]
    for col in missing_cols:
        features[col] = 0

    return features[FEATURE_COLUMNS]


TRAINING_FEATURES, LABELS, SYMPTOM_BINARIZER, SYMPTOM_COLUMNS = build_training_frame()
FEATURE_COLUMNS = TRAINING_FEATURES.columns.tolist()
MODEL = DecisionTreeClassifier(random_state=42)
MODEL.fit(TRAINING_FEATURES, LABELS)


@app.route("/", methods=["GET"])
def home():
    return render_template("Disease.html", result=None, error=None, last_input=None)


@app.route("/predict", methods=["POST"])
def predict():
    form_data = request.form.to_dict()

    try:
        age = int(form_data.get("age", "").strip())
        symptom_count = int(form_data.get("symptom_count", "").strip())
    except (ValueError, AttributeError):
        return render_template(
            "Disease.html",
            error="Age and symptom count must be valid numbers.",
            result=None,
            last_input=form_data,
        )

    symptoms_raw = form_data.get("symptoms", "")
    symptom_tokens = tokenize_symptoms(symptoms_raw)
    if not symptom_tokens:
        return render_template(
            "Disease.html",
            error="Please provide at least one symptom.",
            result=None,
            last_input=form_data,
        )

    gender_text = form_data.get("gender", "other").strip().lower() or "other"
    symptom_count = symptom_count if symptom_count > 0 else len(symptom_tokens)

    features = assemble_features(
        age=age,
        gender=gender_text,
        symptom_count=symptom_count,
        symptoms=symptom_tokens,
    )
    prediction = MODEL.predict(features)[0]

    return render_template(
        "Disease.html",
        result=prediction,
        error=None,
        last_input=form_data,
    )


if __name__ == "__main__":
    app.run(debug=True)

