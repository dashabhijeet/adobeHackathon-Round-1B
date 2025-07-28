import os
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Input/Output paths
input_json_path = "challenge1b_input.json"
output_json_path = "challenge1b_output.json"
collection_dirs = ["Collection 1", "Collection 2", "Collection 3"]

# Load input
with open(input_json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

persona = data["persona"]["role"]
job = data["job_to_be_done"]["task"]
query = f"{persona}: {job}"
query_embedding = model.encode(query, convert_to_tensor=True)

# Gather all headings from extracted JSONs
candidate_headings = []

for doc in data["documents"]:
    filename = doc["filename"]
    extracted_json = None

    for collection in collection_dirs:
        possible_path = os.path.join(collection, "extracted", filename.replace(".pdf", ".json"))
        if os.path.exists(possible_path):
            extracted_json = possible_path
            break

    if not extracted_json:
        print(f"⚠ Skipping: No extracted headings for {filename}")
        continue

    with open(extracted_json, "r", encoding="utf-8") as f:
        content = json.load(f)
        title = content.get("title", "")
        outline = content.get("outline", [])
        for section in outline:
            candidate_headings.append({
                "document": filename,
                "title": title,
                "heading": section["text"],
                "page_number": section["page_number"]
            })

# Rank by semantic similarity
scored = []
for entry in candidate_headings:
    sim_score = util.cos_sim(model.encode(entry["heading"], convert_to_tensor=True), query_embedding).item()
    entry["score"] = sim_score
    scored.append(entry)

# Sort and pick top 5
top_sections = sorted(scored, key=lambda x: -x["score"])[:5]

# Build output
output = {
    "metadata": {
        "input_documents": [doc["filename"] for doc in data["documents"]],
        "persona": persona,
        "job_to_be_done": job,
        "processing_timestamp": datetime.now().isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

for rank, item in enumerate(top_sections, 1):
    output["extracted_sections"].append({
        "document": item["document"],
        "section_title": item["heading"],
        "importance_rank": rank,
        "page_number": item["page_number"]
    })
    output["subsection_analysis"].append({
        "document": item["document"],
        "refined_text": item["heading"],
        "page_number": item["page_number"]
    })

# Save
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("✅ Generated:", output_json_path)