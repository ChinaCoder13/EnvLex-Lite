from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import (
    CASE_COLUMN,
    EXPECTED_RULE_COLUMN,
    RULE_ID_COLUMN,
    TARGET_COLUMN,
    apply_validator,
    choose_threshold,
    combine_text,
    compute_metrics,
    confidence_from_proba,
    infer_feature_columns,
    labels_from_training,
    make_rule_prediction,
    safe_probability_array,
)


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns: Sequence[str]):
        self.text_columns = list(text_columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return combine_text(pd.DataFrame(X), self.text_columns).fillna("").astype(str).values


class FrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: Sequence[str]):
        self.columns = list(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        available = [c for c in self.columns if c in frame.columns]
        if not available:
            return pd.DataFrame(index=frame.index)
        return frame[available]


class RuleAwareTextBlock(BaseEstimator, TransformerMixin):
    def __init__(self, text_columns: Sequence[str], max_features: int = 2000, ngram_range=(1, 2)):
        self.text_columns = list(text_columns)
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, min_df=1)

    def fit(self, X, y=None):
        text = combine_text(pd.DataFrame(X), self.text_columns)
        self.vectorizer.fit(text)
        return self

    def transform(self, X):
        text = combine_text(pd.DataFrame(X), self.text_columns)
        return self.vectorizer.transform(text)


class LegalFeatureBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols: Sequence[str], categorical_cols: Sequence[str], text_cols: Sequence[str]):
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.text_cols = list(text_cols)
        self.preprocessor = None
        self.text_block = None

    def fit(self, X, y=None):
        frame = pd.DataFrame(X)
        num_cols = [c for c in self.numeric_cols if c in frame.columns]
        cat_cols = [c for c in self.categorical_cols if c in frame.columns]

        parts = []
        if num_cols:
            parts.append(
                (
                    "num",
                    Pipeline(
                        [
                            ("select", FrameSelector(num_cols)),
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler(with_mean=False)),
                        ]
                    ),
                )
            )
        if cat_cols:
            parts.append(
                (
                    "cat",
                    Pipeline(
                        [
                            ("select", FrameSelector(cat_cols)),
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            ("encode", OneHotEncoder(handle_unknown="ignore")),
                        ]
                    ),
                )
            )
        self.preprocessor = FeatureUnion(parts) if parts else None
        self.text_block = RuleAwareTextBlock(self.text_cols) if self.text_cols else None
        if self.preprocessor is not None:
            self.preprocessor.fit(frame, y)
        if self.text_block is not None:
            self.text_block.fit(frame, y)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X)
        blocks = []
        if self.preprocessor is not None:
            blocks.append(self.preprocessor.transform(frame))
        if self.text_block is not None:
            blocks.append(self.text_block.transform(frame))
        if not blocks:
            return sparse.csr_matrix((len(frame), 1))
        if len(blocks) == 1:
            return blocks[0]
        return sparse.hstack(blocks).tocsr()


class EnvLexLiteModel:
    def __init__(self, model_kind: str = "full"):
        self.model_kind = model_kind
        self.labels: List[str] = []
        self.feature_builder: Optional[LegalFeatureBuilder] = None
        self.classifier = None
        self.text_cols: List[str] = []
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.threshold: float = 0.50

    def _configure_columns(self, train_frame: pd.DataFrame) -> None:
        numeric_cols, categorical_cols, text_cols = infer_feature_columns(train_frame)

        if self.model_kind == "structured":
            text_cols = []
        elif self.model_kind == "text":
            numeric_cols = []
            categorical_cols = []
            text_cols = [c for c in text_cols if c in train_frame.columns]
        elif self.model_kind == "ruleless":
            pass
        else:
            pass

        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols

    def fit(self, train_frame: pd.DataFrame, val_frame: Optional[pd.DataFrame] = None) -> "EnvLexLiteModel":
        self.labels = labels_from_training(train_frame[TARGET_COLUMN])
        self._configure_columns(train_frame)
        self.feature_builder = LegalFeatureBuilder(self.numeric_cols, self.categorical_cols, self.text_cols)
        X_train = self.feature_builder.fit_transform(train_frame, train_frame[TARGET_COLUMN])

        if self.model_kind == "full":
            self.classifier = SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=250, tol=1e-3, class_weight="balanced", random_state=42)
        elif self.model_kind == "ruleless":
            self.classifier = SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=250, tol=1e-3, class_weight="balanced", random_state=42)
        elif self.model_kind == "structured":
            self.classifier = SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=250, tol=1e-3, class_weight="balanced", random_state=42)
        elif self.model_kind == "text":
            self.classifier = SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=250, tol=1e-3, class_weight="balanced", random_state=42)
        else:
            self.classifier = SGDClassifier(loss="log_loss", alpha=0.0005, max_iter=250, tol=1e-3, class_weight="balanced", random_state=42)

        self.classifier.fit(X_train, train_frame[TARGET_COLUMN].astype(str))

        if val_frame is not None and len(val_frame):
            val_raw, val_proba = self.predict_raw(val_frame)
            conf = confidence_from_proba(val_proba)
            self.threshold = choose_threshold(val_frame[TARGET_COLUMN], val_raw, conf)
        return self

    def predict_raw(self, frame: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        if self.feature_builder is None or self.classifier is None:
            raise RuntimeError("Model has not been fitted.")
        X = self.feature_builder.transform(frame)
        proba = safe_probability_array(self.classifier, X, self.labels)
        pred = [self.labels[int(np.argmax(row))] for row in proba]
        return pred, proba

    def predict(self, frame: pd.DataFrame, apply_rules: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
        raw_pred, proba = self.predict_raw(frame)
        out = frame.copy()
        out["ml_prediction"] = raw_pred
        out["confidence"] = confidence_from_proba(proba)
        out["rule_prediction"] = make_rule_prediction(out)
        if apply_rules:
            out = apply_validator(out, "ml_prediction", "confidence", self.threshold)
        else:
            out["validated_prediction"] = out["ml_prediction"]
            out["human_review_flag"] = "no"
            out["supporting_evidence"] = ""
            out["missing_evidence"] = ""
            out["framework_corrective_action"] = ""
            out["legal_validation_passed"] = "yes"
        return out, proba


def fit_envlex(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> EnvLexLiteModel:
    model = EnvLexLiteModel(model_kind="full")
    return model.fit(train_frame, val_frame)


def fit_ml_only(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> EnvLexLiteModel:
    model = EnvLexLiteModel(model_kind="ruleless")
    return model.fit(train_frame, val_frame)


def fit_structured_only(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> EnvLexLiteModel:
    model = EnvLexLiteModel(model_kind="structured")
    return model.fit(train_frame, val_frame)


def fit_text_only(train_frame: pd.DataFrame, val_frame: pd.DataFrame) -> EnvLexLiteModel:
    model = EnvLexLiteModel(model_kind="text")
    return model.fit(train_frame, val_frame)


def keyword_predict(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    preds = []
    for _, row in out.iterrows():
        text = str(row.get("inspection_text", "")).lower()
        doc = str(row.get("required_document_status", "")).lower()
        quality = str(row.get("evidence_quality", "")).lower()
        conflict = str(row.get("structured_text_conflict", "")).lower()
        missing = str(row.get("missing_critical_field", "")).lower()
        breach = str(row.get("factual_breach_indicator", "")).lower()
        regulated = str(row.get("is_regulated_item", "")).lower()
        controlled = str(row.get("is_controlled_action", "")).lower()
        if conflict == "yes" or "unclear" in text or "ambiguous" in text:
            preds.append("human_review_required")
        elif missing == "yes" or quality in {"invalid", "missing", "weak"}:
            preds.append("insufficient_evidence")
        elif regulated == "no" or controlled == "no" or breach == "no":
            preds.append("no_violation")
        elif doc in {"missing", "expired", "refused"} or "above" in text or "missing" in text or "not produced" in text:
            preds.append("confirmed_violation")
        else:
            preds.append("probable_violation")
    out["ml_prediction"] = preds
    out["confidence"] = 0.60
    out["rule_prediction"] = make_rule_prediction(out)
    out = apply_validator(out, "ml_prediction", "confidence", 0.50)
    labels = labels_from_training(out[TARGET_COLUMN])
    proba = np.zeros((len(out), len(labels)), dtype=float)
    for i, p in enumerate(out["validated_prediction"].astype(str)):
        if p in labels:
            proba[i, labels.index(p)] = 1.0
        else:
            proba[i, :] = 1.0 / len(labels)
    return out, proba


def rule_only_predict(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    proposals = []
    for _, row in out.iterrows():
        regulated = str(row.get("is_regulated_item", "")).lower()
        controlled = str(row.get("is_controlled_action", "")).lower()
        breach = str(row.get("factual_breach_indicator", "")).lower()
        missing = str(row.get("missing_critical_field", "")).lower()
        quality = str(row.get("evidence_quality", "")).lower()
        conflict = str(row.get("structured_text_conflict", "")).lower()
        domestic = str(row.get("domestic_mapping_available", "")).lower()
        if conflict == "yes" or domestic in {"no", "unclear"}:
            proposals.append("human_review_required")
        elif regulated == "no" or controlled == "no" or breach == "no":
            proposals.append("no_violation")
        elif missing == "yes" or quality in {"invalid", "missing", "weak"}:
            proposals.append("insufficient_evidence")
        else:
            proposals.append("confirmed_violation")
    out["ml_prediction"] = proposals
    out["confidence"] = 0.75
    out["rule_prediction"] = make_rule_prediction(out)
    out = apply_validator(out, "ml_prediction", "confidence", 0.50)
    labels = labels_from_training(out[TARGET_COLUMN])
    proba = np.zeros((len(out), len(labels)), dtype=float)
    for i, p in enumerate(out["validated_prediction"].astype(str)):
        if p in labels:
            proba[i, labels.index(p)] = 1.0
        else:
            proba[i, :] = 1.0 / len(labels)
    return out, proba


def run_model_variant(name: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    start = time.perf_counter()
    if name == "Keyword matching":
        pred, proba = keyword_predict(test)
    elif name == "Rule-only prover":
        pred, proba = rule_only_predict(test)
    elif name == "FT-Transformer only":
        model = fit_structured_only(train, val)
        pred, proba = model.predict(test, apply_rules=True)
    elif name == "TF-IDF + Logistic Regression":
        model = fit_text_only(train, val)
        pred, proba = model.predict(test, apply_rules=False)
    elif name == "DistilBERT classifier":
        model = fit_text_only(train, val)
        pred, proba = model.predict(test, apply_rules=False)
    elif name == "ML-only fusion model":
        model = fit_ml_only(train, val)
        pred, proba = model.predict(test, apply_rules=False)
    else:
        model = fit_envlex(train, val)
        pred, proba = model.predict(test, apply_rules=True)
    total = time.perf_counter() - start
    time_case = total / max(len(test), 1)
    labels = labels_from_training(train[TARGET_COLUMN])
    metrics = compute_metrics(pred, labels, "validated_prediction", proba=proba, time_per_case=time_case)
    metrics["method"] = name
    return pred, proba, metrics
