import os
import fitz
import json
import joblib
import re
from collections import Counter

DATE_KEYWORDS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "noon", "midnight", "am", "pm"
]

def extract_lines_with_features(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    for page_num in range(len(doc)):
        blocks = doc[page_num].get_text("dict")["blocks"]
        for block in blocks:
            for line in block.get("lines", []):
                full_text = ""
                bold = False
                font_size = 0
                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if text:
                        full_text += text + " "
                        if "bold" in span["font"].lower():
                            bold = True
                        font_size = max(font_size, span["size"])
                full_text = full_text.strip()
                if full_text:
                    lines.append({
                        "text": full_text,
                        "font_size": font_size,
                        "bold": bold,
                        "page_number": page_num + 1
                    })
    return lines

def is_date_heading(text):
    lower = text.lower()
    return any(word in lower for word in DATE_KEYWORDS) or re.search(r'\b\d{4}\b', lower)

def merge_fragmented_headings(lines):
    merged = []
    skip_next = False
    for i, line in enumerate(lines):
        if skip_next:
            skip_next = False
            continue
        if i + 1 < len(lines):
            curr, next_line = line["text"], lines[i+1]["text"]
            if re.match(r"^\d+[\.\)]?$", curr.strip()) and next_line.strip().istitle():
                merged_line = {
                    **line,
                    "text": f"{curr.strip()} {next_line.strip()}",
                    "is_heading": True
                }
                merged.append(merged_line)
                skip_next = True
                continue
        merged.append(line)
    return merged

def extract_title(lines):
    first_page_lines = [l for l in lines if l["page_number"] == 1]
    if not first_page_lines:
        return ""
    max_font = max(l["font_size"] for l in first_page_lines)
    title_lines = [
        l["text"] for l in first_page_lines
        if abs(l["font_size"] - max_font) < 0.5
    ]
    title = " ".join(title_lines)
    return re.sub(r"\s+", " ", title).strip()

def determine_heading_levels(predicted_lines):
    font_sizes = [line["font_size"] for line in predicted_lines if line["is_heading"]]
    if not font_sizes:
        return []
    common_sizes = Counter(font_sizes).most_common()
    size_to_level = {}
    if len(common_sizes) >= 1:
        size_to_level[common_sizes[0][0]] = "H1"
    if len(common_sizes) >= 2:
        size_to_level[common_sizes[1][0]] = "H2"
    if len(common_sizes) >= 3:
        size_to_level[common_sizes[2][0]] = "H3"

    outline = []
    for line in predicted_lines:
        if line.get("is_heading") and not is_date_heading(line["text"]):
            heading_level = size_to_level.get(line["font_size"])
            if heading_level:
                outline.append({
                    "level": heading_level,
                    "text": line["text"],
                    "page_number": line["page_number"]
                })
    return outline

def predict(pdf_path, output_path, model):
    lines = extract_lines_with_features(pdf_path)
    predictions = []
    for line in lines:
        features = {"font_size": line["font_size"], "bold": line["bold"]}
        is_heading = model.predict([features])[0]
        line["is_heading"] = bool(is_heading)
        predictions.append(line)
    predictions = merge_fragmented_headings(predictions)

    structured_output = {
        "title": extract_title(lines),
        "outline": determine_heading_levels(predictions)
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured_output, f, indent=2)

    print(f"✅ {os.path.basename(pdf_path)} processed")
    print("📄 Title:", structured_output["title"])
    print("📚 Headings found:", len(structured_output["outline"]))

def run_on_directory(pdf_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model = joblib.load("heading_classifier/heading_classifier.joblib")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_folder, filename)
        output_path = os.path.join(output_folder, filename.replace(".pdf", ".json"))
        predict(pdf_path, output_path, model)

# Example usage
if __name__ == "__main__":
    run_on_directory(
        pdf_folder="Collection 3/PDFs",
        output_folder="Collection 3/extracted"
    )