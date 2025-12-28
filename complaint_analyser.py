# =====================================================
# CitySense360 - Citizen Complaint Analyzer
# NLP + LLM-style Classification, Summarization & Routing
# Single-file implementation
# =====================================================

from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# -----------------------------------------------------
# 1. LOAD MODELS
# -----------------------------------------------------
print("Loading NLP models...")

# Zero-shot classifier
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Summarization model
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn"
)

# Embedding model (for routing logic if needed)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------------------------------
# 2. DEFINE CATEGORIES & DEPARTMENTS
# -----------------------------------------------------
CATEGORIES = {
    "Traffic Issue": "Department of Traffic Management",
    "Water Supply Issue": "Water & Sanitation Department",
    "Electricity Issue": "Electricity Board",
    "Garbage / Sanitation": "Municipal Sanitation Department",
    "Public Safety": "Police & Public Safety Department",
    "Road Infrastructure": "Public Works Department",
    "Environmental Issue": "Environmental Protection Department",
    "Other": "General Municipal Office"
}

LABELS = list(CATEGORIES.keys())

# -----------------------------------------------------
# 3. CORE ANALYSIS FUNCTION
# -----------------------------------------------------
def analyze_complaint(complaint_text):
    # ---- Classification ----
    classification = classifier(
        complaint_text,
        candidate_labels=LABELS
    )

    predicted_category = classification["labels"][0]
    confidence = classification["scores"][0]

    # ---- Summarization ----
    summary = summarizer(
        complaint_text,
        max_length=60,
        min_length=20,
        do_sample=False
    )[0]["summary_text"]

    # ---- Routing ----
    department = CATEGORIES.get(predicted_category, "General Municipal Office")

    return {
        "original_complaint": complaint_text,
        "summary": summary,
        "category": predicted_category,
        "confidence": round(confidence, 2),
        "assigned_department": department
    }

# -----------------------------------------------------
# 4. INTERACTIVE TEST
# -----------------------------------------------------
if __name__ == "__main__":
    print("\nCitySense360 – Citizen Complaint Analyzer")
    print("----------------------------------------")

    while True:
        complaint = input("\nEnter a citizen complaint (or type 'exit'): ")
        if complaint.lower() == "exit":
            break

        result = analyze_complaint(complaint)

        print("\n--- Analysis Result ---")
        print("Summary              :", result["summary"])
        print("Category             :", result["category"])
        print("Confidence            :", result["confidence"])
        print("Routed To Department :", result["assigned_department"])
