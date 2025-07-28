# 🏆 Adobe India Hackathon – Round 1B Submission

## 🔧 Challenge ID: round_1b_002  
*Test Case Name:* travel_planner  
*Description:* France Travel

---

## 📌 Objective

Create a system that:
- Extracts *headings and structure* from each PDF using a trained model.
- Generates a *semantic summarization* based on a provided *persona* and *job to be done*.

---

## 🧠 My Approach

### ✅ Step 1: Heading Detection (ML-based)

I trained a RandomForestClassifier using labeled heading data.

#### Features used per line:
| Feature              | Purpose                                                   |
|----------------------|-----------------------------------------------------------|
| font_size          | Headings usually have larger font                        |
| bold               | Often bolded, but not always reliable on its own         |
| starts_with_bullet | Bullet points sometimes indicate headings                |
| ends_with_punctuation | Headings usually don’t end with punctuation          |
| line_length        | Headings tend to be short                                 |
| is_line_alone      | Headings often appear on a separate line                  |
| contains_numbers   | Step-based headings (e.g., "1. Step One")                 |

#### Post-processing:

- Merges fragmented headings like:

3. 

Overview → 3. Overview

- Removes false positives:
- Dates, weekdays, "Noon", etc.
- Lines on *page 1* likely to be *title*, not headings
- Lowercase-starting lines after headings with - or : suffixes

---

### ✅ Step 2: Summarization with Semantic Matching

I use *Sentence-BERT* to embed:
- The full task prompt:  
"Travel Planner: Plan a trip of 4 days for a group of 10 college friends."
- All headings from the PDFs (stored under extracted/)

I compute *cosine similarity* between them, and select the *top 5 most relevant headings*, ranked by importance.

---

## 📁 Directory Structure

Challenge_1b/ │ ├── Collection 1/ │   └── PDFs/ │   └── extracted/        # Extracted titles + headings (.json) │ ├── Collection 2/ │   └── PDFs/ │   └── extracted/ │ ├── Collection 3/ │   └── PDFs/ │   └── extracted/ │ ├── heading_classifier.joblib       # Trained heading classifier ├── challenge1b_input.json          # Summarization input prompt ├── challenge1b_output.json         # Final output (after summarization) │ ├── predict_headings_all.py        # Extracts headings from all PDFs ├── summarize_collection.py        # Performs semantic summarization └── README.md

---

## 💻 How to Run

### 📦 Install requirements

```bash
pip install -r requirements.txt

PyMuPDF
scikit-learn
joblib
sentence-transformers


---

🧪 Step 1: Extract Headings from PDFs

python predict_headings_all.py

This will read all PDFs from Collection {1,2,3}/PDFs

It will generate extracted/*.json in each respective folder, with:

{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Section Heading",
      "page_number": 2
    },
    ...
  ]
}



---

🧠 Step 2: Generate Final Summarization

python summarize_collection.py

This will:

Load challenge1b_input.json

Read relevant headings from extracted files

Select top 5 most relevant based on similarity

Save to challenge1b_output.json




---

📥 Sample Input: challenge1b_input.json

{
  "challenge_info": {
    "challenge_id": "round_1b_002",
    "test_case_name": "travel_planner",