"""
AI Grievance Classification Engine — Web API wrapper.

This turns ai_engine_gemini.py into a small local web service your website's
backend (or frontend, for local testing) can call automatically, instead of
running the script by hand in a terminal.

SETUP
-----
1. pip install flask google-generativeai flask-cors
2. Set your key (same as before):
     $env:GOOGLE_API_KEY = "your-key-here"     (PowerShell)
3. Run:
     python api_server.py
   It will start a local server at http://127.0.0.1:5000

HOW YOUR WEBSITE CALLS IT
--------------------------
Send a POST request to http://127.0.0.1:5000/classify with JSON body:
   {"complaint_text": "There is no water supply in my area for 5 days."}

You'll get back JSON like:
   {
     "category": "Water Supply",
     "severity": "Critical",
     "confidence": 0.97,
     "department": "Municipal Water Board",
     "duplicate": false,
     "duplicate_match_id": null,
     "route": "auto"
   }

Note the extra "route" field: "auto" if confidence >= 0.75, otherwise
"human_review" — this is the "Is Routing Clear?" decision from your workflow
diagram, computed automatically from the AI's own confidence score.

EXAMPLE FROM A BROWSER / FRONTEND (JavaScript)
------------------------------------------------
fetch("http://127.0.0.1:5000/classify", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({complaint_text: "No water supply for 5 days"})
})
  .then(res => res.json())
  .then(data => console.log(data));

This same pattern works no matter what your teammate builds the website
frontend in (plain HTML/JS, React, etc.) — it's just a normal web request.
"""

import os
import sys
import json

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    import google.generativeai as genai
except ImportError:
    print("Missing dependency. Run: pip install flask flask-cors google-generativeai")
    sys.exit(1)

MODEL = "gemini-3.6-flash"
AUTO_ROUTE_CONFIDENCE_THRESHOLD = 0.75

CATEGORIES = {
    "Roads & Infrastructure": "Potholes, damaged roads/footpaths, bridges, street lighting, drainage structures tied to roadworks.",
    "Water Supply": "Drinking water availability, pressure, quality/contamination, tanker service, pipeline faults.",
    "Sanitation & Waste Management": "Garbage collection, public toilets, sewage/drain blockages, waste segregation.",
    "Electricity & Power": "Outages, voltage issues, faulty transformers/lines, billing/metering.",
    "Public Safety & Law and Order": "Crime, harassment, traffic enforcement, stray-animal safety incidents, policing gaps.",
    "Healthcare Services": "Public hospitals/clinics, medicine stock-outs, ambulance response, staffing at health centres.",
    "Education Services": "Government school infrastructure, staffing, mid-day meals, learning materials.",
    "Public Transport": "Public bus/rail service reliability, stops/shelters, overcrowding, schedules.",
    "Corruption & Administrative Misconduct": "Bribery, fund misuse, deliberate service stalling, lack of transparency in public processes.",
    "Environmental & Pollution": "Air/water pollution, illegal burning, industrial discharge, noise pollution, tree felling.",
}

SEVERITIES = {
    "Critical": "Immediate threat to life, safety, or health; affects a large number of citizens; or a total outage of an essential service.",
    "High": "Significant disruption to an individual's or a small group's health, safety, or ability to carry out daily activities.",
    "Medium": "Genuine inconvenience or service failure that is not urgent or safety-critical.",
    "Low": "Minor, cosmetic, or suggestion-type issue with no safety or urgency dimension.",
}

DEPARTMENT_MAP = {
    "Roads & Infrastructure": "Public Works Department (PWD)",
    "Water Supply": "Municipal Water Board",
    "Sanitation & Waste Management": "Municipal Corporation - Sanitation Wing",
    "Electricity & Power": "State Electricity Board",
    "Public Safety & Law and Order": "Police Department",
    "Healthcare Services": "Department of Health & Family Welfare",
    "Education Services": "Department of School Education",
    "Public Transport": "State Transport Authority",
    "Corruption & Administrative Misconduct": "Vigilance & Anti-Corruption Bureau",
    "Environmental & Pollution": "Pollution Control Board",
}

# In-memory log of complaints seen since the server started, used for duplicate
# detection. NOTE: this resets every time the server restarts. Whoever builds
# the real database (Step 3/4) should replace this with a proper lookup against
# stored complaints -- this in-memory version is only enough for a demo.
SEEN_COMPLAINTS = [
    {"id": "SNX-0003", "text": "Garbage has not been collected in Ward 22, Subhash Nagar for over two weeks. It is rotting, attracting rats, and the smell is unbearable near the school."},
    {"id": "SNX-0012", "text": "There has been a complete power outage in Ward 20, Old Bus Stand Area since yesterday evening affecting the whole street, including a household that depends on an oxygen concentrator."},
    {"id": "SNX-0029", "text": "Road near Sector 8, New Market Road has several small potholes that are getting worse with each rain. Kindly schedule repair work."},
    {"id": "SNX-0034", "text": "Water supply in Ward 11, Periyar Nagar has been irregular for over a week, coming only for 20 minutes every alternate day."},
    {"id": "SNX-0084", "text": "A building inspector allegedly took a bribe to overlook safety violations in a construction site near Sector 8, New Market Road."},
]


def build_system_prompt():
    cat_list = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    sev_list = "\n".join(f"- {name}: {desc}" for name, desc in SEVERITIES.items())
    existing_list = "\n".join(f"[{c['id']}] {c['text']}" for c in SEEN_COMPLAINTS[-20:])  # cap context size
    return f"""You are the AI classification engine for a citizen grievance routing platform.

Classify the citizen complaint into exactly one of these 10 categories:
{cat_list}

Assign exactly one severity level:
{sev_list}

Rules for category and severity:
1. Root cause over symptom: classify by whichever department can actually fix the underlying problem, not the department associated with the visible effect (e.g. illness caused by contaminated water is Water Supply, not Healthcare).
2. Safety impact sets a severity floor: if any part of the complaint describes an immediate safety or life risk, severity is at least Critical, regardless of category.
3. Corruption or misconduct allegations are capped at Medium severity unless there is a concurrent, confirmed safety hazard.

Also check whether this complaint is a near-duplicate (same underlying issue, possibly reworded) of any of these already-logged complaints:
{existing_list}

Respond with ONLY a single JSON object, no markdown, no code fences, no extra text, in exactly this shape:
{{"category": "<one of the 10 category names exactly as written above>", "severity": "<Critical|High|Medium|Low>", "confidence": <number between 0 and 1>, "duplicate": <true or false>, "duplicate_match_id": "<matching id from the list above, or null>"}}"""


def classify_complaint(model, complaint_text):
    """Core classification function. This is what Step 3/4 ultimately calls,
    either through this HTTP API or by importing this function directly if
    the website backend is also written in Python."""
    full_prompt = build_system_prompt() + "\n\nComplaint:\n" + complaint_text
    response = model.generate_content(
        full_prompt,
        generation_config={"response_mime_type": "application/json"},
    )
    cleaned = response.text.strip().replace("```json", "").replace("```", "").strip()
    result = json.loads(cleaned)
    result["department"] = DEPARTMENT_MAP.get(result.get("category"), "Unassigned")
    result["route"] = "auto" if result.get("confidence", 0) >= AUTO_ROUTE_CONFIDENCE_THRESHOLD else "human_review"
    return result


app = Flask(__name__)
CORS(app)  # allows a website running on a different port/domain to call this API

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY environment variable is not set.")
    print("Get a key from https://aistudio.google.com/app/apikey and set it before running this server.")
    sys.exit(1)

genai.configure(api_key=api_key)
model = genai.GenerativeModel(MODEL)


@app.route("/classify", methods=["POST"])
def classify_endpoint():
    data = request.get_json(force=True, silent=True) or {}
    complaint_text = data.get("complaint_text", "").strip()

    if not complaint_text:
        return jsonify({"error": "complaint_text is required"}), 400

    try:
        result = classify_complaint(model, complaint_text)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Log this complaint so future submissions can be checked against it.
    new_id = f"SNX-{len(SEEN_COMPLAINTS) + 1:04d}"
    SEEN_COMPLAINTS.append({"id": new_id, "text": complaint_text})
    result["complaint_id"] = new_id

    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    """Simple check so a teammate can confirm the server is running."""
    return jsonify({"status": "ok", "model": MODEL, "complaints_logged": len(SEEN_COMPLAINTS)})


if __name__ == "__main__":
    print(f"AI Engine API starting on http://127.0.0.1:5000")
    print(f"Test it: POST http://127.0.0.1:5000/classify  with body {{\"complaint_text\": \"...\"}}")
    print(f"Health check: GET http://127.0.0.1:5000/health")
    app.run(debug=True, port=5000)
