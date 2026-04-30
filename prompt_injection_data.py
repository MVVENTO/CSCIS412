from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline


DEFAULT_KAGGLE_DIR = Path("data/kaggle_prompt_injection")

TEXT_COLUMNS = ("text", "prompt", "Prompt", "instruction", "input", "content", "question")
LABEL_COLUMNS = ("label", "Label", "class", "Class", "target", "Target", "type", "Type")
MALICIOUS_LABELS = {
    "1",
    "attack",
    "attacker",
    "harmful",
    "injection",
    "jailbreak",
    "malicious",
    "pi",
    "unsafe",
}
SAFE_LABELS = {
    "0",
    "benign",
    "clean",
    "harmless",
    "normal",
    "safe",
}

DEFAULT_SAFE_PROMPTS = [
    "Write a professional email to my professor asking for a deadline extension.",
    "Summarize this public research article in plain language.",
    "Explain how to secure a home Wi-Fi network for a beginner.",
    "Explain how to protect a home Wi-Fi network from common threats.",
    "What are the best ways to prevent phishing attacks at work?",
    "Create a cybersecurity awareness checklist for employees.",
    "Explain multi-factor authentication in simple terms.",
    "Help me write a small-business password policy.",
    "Summarize beginner-friendly NIST cybersecurity best practices.",
    "Create a study guide for networking basics.",
    "Draft a polite follow-up message after a job interview.",
    "Rewrite this paragraph to sound more formal.",
    "Translate this classroom announcement into Spanish.",
    "Compare cloud storage and external hard drives for backups.",
    "Outline a presentation about renewable energy.",
    "Generate interview questions for an entry-level software engineer.",
    "Explain phishing awareness training in simple terms.",
    "Create a grocery list for healthy meals this week.",
    "Help me plan a study schedule for my exams.",
    "Summarize the main ideas of the U.S. Constitution.",
    "Write a short thank-you note to a mentor.",
    "Show me how to open my PC case safely to clean the fans.",
    "Show me how to troubleshoot a computer that will not turn on.",
    "Walk me through reinstalling Windows on my PC.",
    "Help me fix my PC after it stopped booting.",
    "Explain how to enter BIOS on my own computer.",
    "How do I clean dust out of my computer safely?",
    "Show me how to replace a laptop battery step by step.",
    "Walk me through opening my PC so I can install more RAM.",
    "How do I create a bootable Windows USB for my own PC?",
    "Explain how to back up files before repairing my computer.",
    "How can I improve my cybersecurity skills?",
    "How can I learn cybersecurity as a beginner?",
    "What cybersecurity certifications should I study for?",
    "Teach me defensive security best practices.",
    "How do I protect my home network from attackers?",
    "Explain defensive cybersecurity in simple terms.",
    "Create a study plan for learning cybersecurity.",
    "What topics should I study to start a cybersecurity career?",
    "Give me a beginner roadmap for cybersecurity learning.",
    "How do I practice cybersecurity safely and legally?",
    "Explain the difference between offensive and defensive security.",
    "How can I build defensive security skills for IT work?",
    "Summarize cybersecurity fundamentals for beginners.",
    "Recommend legal home lab ideas for learning cybersecurity.",
    "How do I improve my network security skills?",
    "What are the best defensive cybersecurity projects for students?",
    "Teach me about phishing prevention and account security.",
    "How can I secure my own PC and home network?",
    "Create a checklist for improving computer security skills.",
    "What are good books for learning cybersecurity basics?",
    "Explain vulnerability management for beginners.",
    "How do I get better at incident response?",
    "Show me safe ways to practice packet analysis.",
    "Teach me how firewalls and IDS tools work.",
]

SAFE_TOPICS = [
    "algebra homework",
    "American history",
    "beginner Python programming",
    "biology class notes",
    "book club discussion questions",
    "budgeting for college students",
    "business meeting agendas",
    "career development goals",
    "cybersecurity basics",
    "cybersecurity career planning",
    "cybersecurity certifications",
    "cybersecurity skills",
    "classroom attendance policy",
    "climate change research",
    "customer service best practices",
    "data privacy basics",
    "digital note-taking tips",
    "email etiquette",
    "endpoint security basics",
    "ethical hacking in legal learning environments",
    "exercise plans for beginners",
    "financial literacy basics",
    "healthy breakfast recipes",
    "high school chemistry",
    "history essay planning",
    "how to back up family photos",
    "how to prepare for a math quiz",
    "human resources onboarding",
    "intro to databases",
    "job interview preparation",
    "learning Spanish vocabulary",
    "meeting minutes",
    "mental wellness habits",
    "middle school science",
    "multi-factor authentication setup",
    "networking fundamentals",
    "network defense fundamentals",
    "note summarization",
    "office communication",
    "personal productivity",
    "phishing prevention",
    "polite customer replies",
    "presentation skills",
    "project planning",
    "public speaking tips",
    "research paper outlines",
    "resume writing",
    "road trip packing lists",
    "school fundraiser ideas",
    "secure home Wi-Fi settings",
    "secure password habits",
    "security awareness training",
    "small business bookkeeping",
    "software testing basics",
    "software patch management",
    "study habits",
    "threat detection basics",
    "teacher appreciation messages",
    "time management",
    "travel itineraries",
    "using a password manager",
    "website accessibility",
    "workplace collaboration",
    "writing a cover letter",
]

SAFE_ACTIONS = [
    "open a desktop PC case safely",
    "clean dust from a desktop PC",
    "replace RAM in a personal computer",
    "replace a laptop battery",
    "install an SSD in a personal computer",
    "troubleshoot a PC that will not boot",
    "repair a corrupted Windows installation",
    "enter BIOS settings on a home computer",
    "run Windows startup repair",
    "create a Windows recovery USB",
    "back up files before repairing a computer",
    "fix an overheating desktop PC",
    "diagnose a failing hard drive",
    "reset a PC after a bad software update",
    "secure a home router",
    "remove malware from a personal laptop",
]

SAFE_TEMPLATES = [
    "Summarize {topic}.",
    "Explain {topic} in simple terms.",
    "Create a study guide for {topic}.",
    "Write a professional email about {topic}.",
    "Draft a short report about {topic}.",
    "Outline a presentation on {topic}.",
    "Brainstorm ideas related to {topic}.",
    "Give me beginner-friendly advice about {topic}.",
    "Compare two common approaches to {topic}.",
    "Rewrite these notes about {topic} so they sound more formal.",
]

SAFE_ACTION_TEMPLATES = [
    "Show me how to {action}.",
    "Walk me through how to {action}.",
    "How do I {action}?",
    "Help me {action}.",
    "Give me step-by-step instructions to {action}.",
]

SAFE_SUFFIXES = [
    "",
    " Use bullet points.",
    " Keep it short and easy to understand.",
]


@dataclass
class TrainingArtifacts:
    dataset: pd.DataFrame
    model: Pipeline
    accuracy: float
    report: str
    label_counts: dict[str, int]
    # added 
    X_test: pd.Series
    y_test: pd.Series


def _pick_first_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in frame.columns:
            return name
    return None


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _find_existing_directory(candidates: Iterable[str | Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate)
        if path.exists():
            return path
    return None


def _read_supported_columns(csv_path: Path) -> pd.DataFrame:
    header = pd.read_csv(csv_path, nrows=0)
    text_column = _pick_first_column(header, TEXT_COLUMNS)
    if text_column is None:
        raise ValueError(f"{csv_path.name} does not contain a supported text column.")

    label_column = _pick_first_column(header, LABEL_COLUMNS)
    usecols = [text_column]
    if label_column is not None and label_column != text_column:
        usecols.append(label_column)
    return pd.read_csv(csv_path, usecols=usecols)


def normalize_label(value: object) -> str | None:
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return "malicious" if value else "safe"
    if isinstance(value, (int, float)) and value in (0, 1):
        return "malicious" if int(value) == 1 else "safe"

    normalized = _normalize_whitespace(str(value)).lower()
    if normalized in MALICIOUS_LABELS:
        return "malicious"
    if normalized in SAFE_LABELS:
        return "safe"
    return None


def normalize_prompt_frame(
    frame: pd.DataFrame,
    *,
    source_name: str,
    default_label: str | None = None,
) -> pd.DataFrame:
    text_column = _pick_first_column(frame, TEXT_COLUMNS)
    if text_column is None:
        raise ValueError(f"{source_name} does not contain a supported text column.")

    label_column = _pick_first_column(frame, LABEL_COLUMNS)
    normalized = pd.DataFrame(
        {
            "text": frame[text_column].astype(str).map(_normalize_whitespace),
            "source": source_name,
        }
    )

    if label_column is not None:
        normalized["label"] = frame[label_column].map(normalize_label)
    elif default_label is not None:
        normalized["label"] = default_label
    else:
        raise ValueError(
            f"{source_name} does not contain a supported label column and no default label was provided."
        )

    normalized = normalized.replace({"text": {"": pd.NA}})
    normalized = normalized.dropna(subset=["text", "label"]).copy()
    normalized["text_key"] = normalized["text"].str.lower()
    normalized = normalized.drop_duplicates(subset=["text_key"]).drop(columns=["text_key"])
    return normalized.reset_index(drop=True)


def generate_default_safe_prompts() -> list[str]:
    prompts = {_normalize_whitespace(prompt) for prompt in DEFAULT_SAFE_PROMPTS}
    for template in SAFE_TEMPLATES:
        for topic in SAFE_TOPICS:
            for suffix in SAFE_SUFFIXES:
                prompts.add(_normalize_whitespace(template.format(topic=topic) + suffix))
    for template in SAFE_ACTION_TEMPLATES:
        for action in SAFE_ACTIONS:
            prompts.add(_normalize_whitespace(template.format(action=action)))
    return sorted(prompts)


def load_huggingface_prompt_injection_dataset() -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset("neuralchemy/Prompt-injection-dataset", "core")
    frames = []
    for split_name, split in dataset.items():
        split_frame = split.to_pandas()
        frames.append(
            normalize_prompt_frame(split_frame, source_name=f"neuralchemy:{split_name}")
        )
    return pd.concat(frames, ignore_index=True)


def load_kaggle_prompt_injection_in_the_wild(
    download_dir: str | Path | None = None,
    *,
    allow_download: bool = False,
) -> pd.DataFrame:
    root = _find_existing_directory([download_dir, DEFAULT_KAGGLE_DIR])

    if root is None:
        if not allow_download:
            raise FileNotFoundError(
                "The Kaggle prompt-injection dataset was not found locally. "
                f"Expected it under {DEFAULT_KAGGLE_DIR}."
            )

        import kagglehub

        root = Path(
            kagglehub.dataset_download(
                "arielzilber/prompt-injection-in-the-wild",
                output_dir=str(download_dir) if download_dir else None,
            )
        )

    csv_files = sorted(path for path in root.rglob("*.csv") if path.is_file())
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in {root}.")

    frames = []
    for csv_path in csv_files:
        csv_frame = _read_supported_columns(csv_path)
        frames.append(
            normalize_prompt_frame(
                csv_frame,
                source_name=f"kaggle:{csv_path.stem}",
                default_label="malicious",
            )
        )
    return pd.concat(frames, ignore_index=True)


def build_training_dataset(
    *,
    include_huggingface: bool = False,
    include_kaggle: bool = True,
    kaggle_download_dir: str | Path | None = None,
    custom_safe_prompts: Iterable[str] | None = None,
    include_default_safe_prompts: bool = True,
    ignore_source_errors: bool = True,
) -> pd.DataFrame:
    frames = []
    source_errors: list[str] = []

    if include_huggingface:
        try:
            frames.append(load_huggingface_prompt_injection_dataset())
        except Exception as exc:
            if not ignore_source_errors:
                raise
            source_errors.append(f"Hugging Face dataset skipped: {exc}")

    if include_kaggle:
        try:
            frames.append(
                load_kaggle_prompt_injection_in_the_wild(download_dir=kaggle_download_dir)
            )
        except Exception as exc:
            if not ignore_source_errors:
                raise
            source_errors.append(f"Kaggle dataset skipped: {exc}")

    safe_prompts = []
    if include_default_safe_prompts:
        safe_prompts.extend(generate_default_safe_prompts())
    safe_prompts.extend(custom_safe_prompts or [])

    if safe_prompts:
        safe_frame = pd.DataFrame(
            {"text": list(safe_prompts), "label": "safe", "source": "curated_safe"}
        )
        frames.append(normalize_prompt_frame(safe_frame, source_name="curated_safe"))

    if not frames:
        message = "At least one dataset source must be enabled and load successfully."
        if source_errors:
            message = f"{message} Skipped sources: {' | '.join(source_errors)}"
        raise ValueError(message)

    dataset = pd.concat(frames, ignore_index=True)
    dataset["text_key"] = dataset["text"].str.lower()
    dataset = dataset.drop_duplicates(subset=["text_key"]).drop(columns=["text_key"])
    dataset = dataset.reset_index(drop=True)
    dataset.attrs["source_errors"] = source_errors
    return dataset


def rebalance_training_dataset(
    dataset: pd.DataFrame,
    *,
    max_majority_ratio: float = 2.0,
    random_state: int = 42,
) -> pd.DataFrame:
    label_counts = dataset["label"].value_counts()
    if label_counts.empty or len(label_counts) < 2:
        return dataset.reset_index(drop=True)

    minority_size = label_counts.min()
    majority_limit = max(1, int(minority_size * max_majority_ratio))

    frames = []
    for label, group in dataset.groupby("label", group_keys=False):
        if len(group) > majority_limit:
            frames.append(group.sample(n=majority_limit, random_state=random_state))
        else:
            frames.append(group)

    balanced = pd.concat(frames, ignore_index=True)
    return balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def train_prompt_injection_model(
    dataset: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainingArtifacts:
    X_train, X_test, y_train, y_test = train_test_split(
        dataset["text"],
        dataset["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=dataset["label"],
    )

    model = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                strip_accents="unicode",
                                stop_words="english",
                                ngram_range=(1, 3),
                                min_df=2,
                                max_df=0.98,
                                sublinear_tf=True,
                                max_features=40000,
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                lowercase=True,
                                strip_accents="unicode",
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                sublinear_tf=True,
                                max_features=20000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2500,
                    class_weight="balanced",
                    C=1.5,
                    solver="liblinear",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    return TrainingArtifacts(
        dataset=dataset,
        model=model,
        accuracy=accuracy_score(y_test, predictions),
        report=classification_report(y_test, predictions),
        label_counts=dataset["label"].value_counts().to_dict(),
        #added
        X_test=X_test,
        y_test=y_test,
    )

#edited for python
def predict_with_threshold(model: Pipeline, prompt: str, threshold: float = 0.5) -> tuple[str, dict[str, float]]:
    probabilities = model.predict_proba([prompt])[0]

    scores = {
        label: float(score)
        for label, score in zip(model.classes_, probabilities)
    }

    malicious_score = scores.get("malicious", 0.0)

    if malicious_score >= threshold:
        return "malicious", scores

    return "safe", scores
