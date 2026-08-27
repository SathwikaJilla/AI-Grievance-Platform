"""
AI Grievance Classification Engine — Step 2 deliverable (Gemini version).

Reads a citizen complaint, returns category / severity / confidence / duplicate
status as structured JSON, using Google's Gemini API (no training required).

SETUP
-----
1. pip install google-generativeai
2. Get an API key from https://aistudio.google.com/app/apikey
3. Set it as an environment variable before running:
     set GOOGLE_API_KEY=your-key-here          (Windows cmd)
     export GOOGLE_API_KEY="your-key-here"     (Mac/Linux)
4. Run:
     python ai_engine_gemini.py                        -> classifies a couple of sample complaints
     python ai_engine_gemini.py --evaluate              -> runs all cases (may hit free-tier daily quota)
     python ai_engine_gemini.py --evaluate --limit 15   -> runs only the first 15 cases (good for a quota-limited day)
     python ai_engine_gemini.py --evaluate --start 15   -> continues from case 16 onward (e.g. the next day)

   Results are saved to eval_results_cache.json as you go, so you can split a run across
   multiple days without losing progress or re-testing cases you already completed. The
   final accuracy summary always reflects everything completed so far, across all batches.
"""

import os
import sys
import json
import csv
import time
import argparse

try:
    import google.generativeai as genai
    from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, GoogleAPICallError
except ImportError:
    print("Missing dependency. Run: pip install google-generativeai")
    sys.exit(1)

MODEL = "gemini-3.6-flash"

# Rate-limit / retry settings.
# Free tier for gemini-2.5-flash is 5 requests per minute, so we space calls
# out to roughly one every 13 seconds to stay under that limit, and retry
# with backoff if we still get a 429.
MAX_RETRIES = 5
REQUEST_SPACING_SECONDS = 13
FALLBACK_BACKOFF_SECONDS = 10

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

# A small set of "already logged" complaints for the duplicate-detection demo.
EXISTING_COMPLAINTS = [
    {"id": "SNX-0003", "text": "Garbage has not been collected in Ward 22, Subhash Nagar for over two weeks. It is rotting, attracting rats, and the smell is unbearable near the school."},
    {"id": "SNX-0012", "text": "There has been a complete power outage in Ward 20, Old Bus Stand Area since yesterday evening affecting the whole street, including a household that depends on an oxygen concentrator."},
    {"id": "SNX-0029", "text": "Road near Sector 8, New Market Road has several small potholes that are getting worse with each rain. Kindly schedule repair work."},
    {"id": "SNX-0034", "text": "Water supply in Ward 11, Periyar Nagar has been irregular for over a week, coming only for 20 minutes every alternate day."},
    {"id": "SNX-0084", "text": "A building inspector allegedly took a bribe to overlook safety violations in a construction site near Sector 8, New Market Road."},
]


def build_system_prompt():
    cat_list = "\n".join(f"- {name}: {desc}" for name, desc in CATEGORIES.items())
    sev_list = "\n".join(f"- {name}: {desc}" for name, desc in SEVERITIES.items())
    existing_list = "\n".join(f"[{c['id']}] {c['text']}" for c in EXISTING_COMPLAINTS)
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
    """Send one complaint to Gemini and return a parsed structured result.

    Retries on 429 (quota/rate-limit) and transient server errors, honoring
    the server's suggested retry_delay when Gemini provides one.
    """
    full_prompt = build_system_prompt() + "\n\nComplaint:\n" + complaint_text
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = model.generate_content(
                full_prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            cleaned = response.text.strip().replace("```json", "").replace("```", "").strip()
            result = json.loads(cleaned)
            result["department"] = DEPARTMENT_MAP.get(result.get("category"), "Unassigned")
            return result

        except ResourceExhausted as e:
            last_error = e
            wait_seconds = _extract_retry_delay(e) or FALLBACK_BACKOFF_SECONDS
            print(f"  Quota/rate limit hit (attempt {attempt}/{MAX_RETRIES}). Waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

        except (ServiceUnavailable, GoogleAPICallError) as e:
            # A 404 means the model name itself is wrong/retired — retrying won't help, fail fast.
            if "404" in str(e) or "not found" in str(e).lower() or "no longer available" in str(e).lower():
                raise RuntimeError(
                    f"Model '{MODEL}' is not available: {e}\n"
                    f"Check the current model name and update the MODEL constant at the top of this file."
                )
            last_error = e
            wait_seconds = FALLBACK_BACKOFF_SECONDS * attempt
            print(f"  Transient API error (attempt {attempt}/{MAX_RETRIES}): {e}. Waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds)

    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def _extract_retry_delay(error):
    """Pull the suggested retry_delay (in seconds) out of a Gemini 429 error, if present."""
    try:
        for detail in getattr(error, "details", lambda: [])():
            if hasattr(detail, "retry_delay"):
                return detail.retry_delay.seconds + detail.retry_delay.nanos / 1e9
    except Exception:
        pass
    return None


def run_demo(model):
    samples = [
        "There is no water supply in my street for the past 5 days, please send tankers.",
        "A pothole on the main road caused an accident near my house yesterday.",
        "Contaminated water supply has caused several children at the local school to fall sick.",
    ]
    for i, text in enumerate(samples):
        if i > 0:
            time.sleep(REQUEST_SPACING_SECONDS)
        print("\nComplaint:", text)
        try:
            result = classify_complaint(model, text)
            print(json.dumps(result, indent=2))
        except Exception as e:
            print("  Error:", e)


def _load_results_cache(cache_path):
    """Load previously saved per-case results, if any, so batches can resume across runs."""
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_results_cache(cache_path, cache):
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def run_evaluation(model, csv_path="AI_evaluation_cases.csv", start=0, limit=None, cache_path="eval_results_cache.json"):
    """Run the evaluation, optionally on a slice of cases (for splitting across quota-limited days).

    Results for each case are saved to a local JSON cache file as they complete, so you can:
      - run a batch today (e.g. --start 0 --limit 15)
      - run the rest tomorrow (e.g. --start 15)
      - the final accuracy summary always reflects ALL cases completed so far across every run
    """
    if not os.path.exists(csv_path):
        print(f"Could not find {csv_path}. Put this script in the same folder as your dataset files.")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        all_cases = list(csv.DictReader(f))

    total = len(all_cases)
    end = start + limit if limit else total
    batch = all_cases[start:end]

    cache = _load_results_cache(cache_path)

    print(f"Running evaluation on cases {start+1}-{min(end, total)} of {total} (this batch: {len(batch)} cases)...\n")

    for i, case in enumerate(batch, start=start + 1):
        case_id = case["complaint_id"]

        if case_id in cache:
            print(f"[{i}/{total}] {case_id} already done in a previous run, skipping.")
            continue

        if i > start + 1:
            time.sleep(REQUEST_SPACING_SECONDS)

        text = case["complaint_text"]
        try:
            result = classify_complaint(model, text)
        except Exception as e:
            print(f"[{i}/{total}] ERROR: {e}")
            continue

        cat_ok = result["category"] == case["expected_category"]
        sev_ok = result["severity"] == case["expected_severity"]
        dept_ok = result["department"] == case["expected_department"]
        expected_dup = case["expected_is_duplicate"].strip().lower() == "true"
        dup_ok = bool(result.get("duplicate")) == expected_dup

        cache[case_id] = {
            "cat_ok": cat_ok, "sev_ok": sev_ok, "dept_ok": dept_ok, "dup_ok": dup_ok,
            "got_category": result["category"], "got_severity": result["severity"],
            "expected_category": case["expected_category"], "expected_severity": case["expected_severity"],
        }
        _save_results_cache(cache_path, cache)  # save after every case, so a crash/quota-hit loses nothing

        mark = "OK" if cat_ok and sev_ok else "X "
        print(f"[{mark}] {case_id}  expected={case['expected_category']}/{case['expected_severity']}  got={result['category']}/{result['severity']}")

    # Summary is always computed from EVERYTHING saved in the cache so far, across all batches/days.
    completed = [c for c in all_cases if c["complaint_id"] in cache]
    n = len(completed)
    if n == 0:
        print("\nNo cases completed yet.")
        return

    cat_correct = sum(cache[c["complaint_id"]]["cat_ok"] for c in completed)
    sev_correct = sum(cache[c["complaint_id"]]["sev_ok"] for c in completed)
    dept_correct = sum(cache[c["complaint_id"]]["dept_ok"] for c in completed)
    dup_correct = sum(cache[c["complaint_id"]]["dup_ok"] for c in completed)

    print(f"\n--- Accuracy summary ({n}/{total} cases completed so far) ---")
    print(f"Category accuracy:   {round(100 * cat_correct / n)}%")
    print(f"Severity accuracy:   {round(100 * sev_correct / n)}%")
    print(f"Department accuracy: {round(100 * dept_correct / n)}%")
    print(f"Duplicate-flag match:{round(100 * dup_correct / n)}%")

    if n < total:
        print(f"\n{total - n} case(s) remaining. Run again with --start {n} to continue where you left off")
        print(f"(cases already completed are automatically skipped, so --start 0 also works fine).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate", action="store_true", help="Run the accuracy evaluation")
    parser.add_argument("--csv", default="AI_evaluation_cases.csv", help="Path to the evaluation CSV")
    parser.add_argument("--start", type=int, default=0, help="Index of the first case to run in this batch (0-based). Use this to resume where a previous run left off.")
    parser.add_argument("--limit", type=int, default=None, help="How many cases to run in this batch. Omit to run all remaining cases.")
    parser.add_argument("--cache", default="eval_results_cache.json", help="Path to the results cache file (tracks completed cases across batches/days)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY environment variable is not set.")
        print("Get a key from https://aistudio.google.com/app/apikey and set it before running this script.")
        sys.exit(1)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL)

    if args.evaluate:
        run_evaluation(model, args.csv, start=args.start, limit=args.limit, cache_path=args.cache)
    else:
        run_demo(model)
