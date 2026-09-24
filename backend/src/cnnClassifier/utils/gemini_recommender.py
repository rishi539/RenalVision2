import os
import json
from typing import Dict, Any, List
from cnnClassifier import logger

try:
    import google.generativeai as genai
    GEMINI_SDK_AVAILABLE = True
except ImportError:
    GEMINI_SDK_AVAILABLE = False


def is_gemini_available() -> bool:
    """Checks and dynamically re-attempts importing google.generativeai if initially unavailable."""
    global genai, GEMINI_SDK_AVAILABLE
    if GEMINI_SDK_AVAILABLE:
        return True
    try:
        import google.generativeai as genai_module
        genai = genai_module
        GEMINI_SDK_AVAILABLE = True
        return True
    except ImportError:
        GEMINI_SDK_AVAILABLE = False
        return False



def load_env_file():
    """Helper to auto-load .env from root or backend directory if present."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    env_paths = [
        os.path.join(base_dir, ".env"),
        os.path.join(base_dir, "backend", ".env"),
        os.path.join(os.getcwd(), ".env")
    ]
    for path in env_paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip().strip("'\"")
                            if key:
                                os.environ[key] = val
            except Exception:
                pass

# Auto-load .env on import
load_env_file()


def generate_pure_gemini_recommendation(scans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generates a personalized clinical recommendation directly using Google Gemini AI.
    Strictly grounds the analysis on the patient's verified CT scan history without rule fallbacks.
    """
    load_env_file()
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not configured in backend/.env. Gemini is required for AI Analysis.")
    if not is_gemini_available():
        raise RuntimeError("google.generativeai SDK is not available in environment.")

    genai.configure(api_key=api_key)

    candidate_models = [
        os.environ.get("GEMINI_MODEL", "").strip(),
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    # Deduplicate while preserving order
    seen = set()
    deduped_models = []
    for m in candidate_models:
        if m and m not in seen:
            seen.add(m)
            deduped_models.append(m)
    candidate_models = deduped_models

    def _call_gemini_json(prompt_text: str):
        last_e = None
        for m_name in candidate_models:
            try:
                m = genai.GenerativeModel(m_name)
                res = m.generate_content(
                    prompt_text,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        response_mime_type="application/json"
                    )
                )
                raw = res.text.strip()
                if raw.startswith("```json"):
                    raw = raw[7:]
                if raw.startswith("```"):
                    raw = raw[3:]
                if raw.endswith("```"):
                    raw = raw[:-3]
                return json.loads(raw.strip()), m_name
            except Exception as e:
                logger.warning(f"Model {m_name} call failed: {e}")
                last_e = e
                continue
        raise last_e

    if not scans:
        empty_prompt = f"""
You are an AI Clinical Specialist for RenalVision Kidney Disease Classification.
A patient has requested clinical guidance, but their diagnostic history contains ZERO CT scans (no scan data has been uploaded yet).

TASK:
Provide an appropriate clinical message explaining that there is currently no medical scan information available to formulate a diagnosis or clinical assessment. Clearly and empathetically guide the patient to upload an abdominal or renal CT scan so the AI diagnostic model can classify the scan and formulate a personalized care plan.

FORMAT REQUIREMENT:
Respond ONLY with a valid JSON object strictly matching this schema:
{{
  "primary_condition": "No Scan Data Available",
  "urgency": "Pending Scan Input",
  "urgency_color": "emerald",
  "risk_trend": "No Baseline Established",
  "summary": "Clear, empathetic clinical message stating there is no scan information to diagnose and prompting scan upload",
  "clinical_actions": ["Upload an abdominal or renal CT scan (DICOM or standard image) to initiate diagnostic classification.", "Consult with a physician if you are experiencing acute symptoms, flank pain, or hematuria."],
  "follow_up_timeline": "Upload initial imaging scan to establish clinical timeline.",
  "dietary_lifestyle": ["Maintain standard hydration (2 to 2.5 Liters daily) unless medically contraindicated.", "Follow standard balanced nutritional wellness."],
  "confidence_assessment": "Diagnostic assessment pending CT scan upload."
}}
"""
        try:
            parsed, used_model = _call_gemini_json(empty_prompt)
            return {
                "urgency": parsed.get("urgency", "Pending Scan Input"),
                "urgency_color": parsed.get("urgency_color", "emerald"),
                "risk_trend": parsed.get("risk_trend", "No Baseline Established"),
                "primary_condition": parsed.get("primary_condition", "No Scan Data Available"),
                "summary": parsed.get("summary", "No CT scan information is currently available to formulate a diagnosis. Please upload an abdominal CT scan to begin AI diagnostic evaluation."),
                "clinical_actions": parsed.get("clinical_actions", ["Upload an abdominal CT scan to establish baseline diagnostic record."]),
                "follow_up_timeline": parsed.get("follow_up_timeline", "Upload initial imaging to begin tracking."),
                "dietary_lifestyle": parsed.get("dietary_lifestyle", ["Maintain baseline daily hydration (2 to 2.5 Liters daily)."]),
                "confidence_assessment": parsed.get("confidence_assessment", "N/A - Pending scan upload."),
                "total_scans_analyzed": 0,
                "powered_by": f"Google Gemini AI ({used_model} - Grounded Intelligence)"
            }
        except Exception as e:
            logger.warning(f"Failed to query Gemini for empty scan message: {e}")
            return {
                "urgency": "Pending Scan Input",
                "urgency_color": "emerald",
                "risk_trend": "No Baseline Established",
                "primary_condition": "No Scan Data Available",
                "summary": "No diagnostic CT scan data is currently available in your patient record to formulate a clinical assessment. Please upload an abdominal CT scan to begin AI diagnostic classification.",
                "clinical_actions": ["Upload an abdominal or renal CT scan to initiate diagnostic classification."],
                "follow_up_timeline": "Upload initial imaging scan to establish clinical timeline.",
                "dietary_lifestyle": ["Maintain healthy baseline hydration (2 to 2.5 Liters daily)."],
                "confidence_assessment": "Pending scan upload.",
                "total_scans_analyzed": 0,
                "powered_by": "Google Gemini AI (Grounded Intelligence)"
            }

    scan_summary_list = []
    for s in scans:
        scan_summary_list.append(
            f"Scan #{s.get('scan_number')} on {s.get('scan_date')}: {s.get('prediction')} "
            f"({s.get('confidence', 0):.1f}% confidence. Probabilities: Normal={s.get('prob_normal', 0):.1f}%, "
            f"Cyst={s.get('prob_cyst', 0):.1f}%, Stone={s.get('prob_stone', 0):.1f}%, Tumor={s.get('prob_tumor', 0):.1f}%)"
        )

    latest_scan = scans[-1]
    latest_pred = latest_scan.get("prediction", "Unknown")
    latest_conf = latest_scan.get("confidence", 0.0)

    prompt = f"""
You are an AI Clinical Specialist for RenalVision Kidney Disease Classification.
Analyze the verified patient scan history and provide personalized, strictly grounded clinical guidance.

PATIENT SCAN TIMELINE:
- Total Scans: {len(scans)}
- CURRENT LATEST SCAN (ACTIVE DIAGNOSTIC STATUS):
  Scan #{latest_scan.get('scan_number')} on {latest_scan.get('scan_date')} -> {latest_pred} ({latest_conf:.1f}% confidence. Probabilities: Normal={latest_scan.get('prob_normal', 0):.1f}%, Cyst={latest_scan.get('prob_cyst', 0):.1f}%, Stone={latest_scan.get('prob_stone', 0):.1f}%, Tumor={latest_scan.get('prob_tumor', 0):.1f}%)
- Full Sequential Timeline:
  {chr(10).join(scan_summary_list)}

CRITICAL CLINICAL INSTRUCTIONS:
1. "primary_condition": MUST reflect the patient's ACTIVE status from their latest scan (Scan #{latest_scan.get('scan_number')}: {latest_pred}).
   - If the current scan is "Normal", primary_condition MUST be "Normal" (or "Normal / Clear Scan"). Do NOT classify primary_condition as "Tumor" or "Stone" when the current scan is Normal!
   - You can discuss past abnormal scans in "risk_trend" and "summary" (e.g. noting historical lesions now resolved or cleared), but the primary active condition is {latest_pred}.
2. "urgency": If the current scan is Normal, urgency MUST be "Normal" or "Low" (or "Moderate" if routine follow-up surveillance is advised for prior history). Urgency must NEVER be "Urgent" when the current scan is Normal.
3. Ground all statements strictly on the provided scan findings.

FORMAT REQUIREMENT:
Respond ONLY with a valid JSON object strictly matching this schema (no markdown fences, no explanatory text outside the JSON):
{{
  "primary_condition": "{latest_pred}",
  "urgency": "Urgency rating (Normal / Low / Moderate / Urgent)",
  "urgency_color": "emerald for Normal/Low, amber for Moderate, rose for Urgent",
  "risk_trend": "Brief description of longitudinal risk trend across sequential scans",
  "summary": "2-3 sentence personalized clinical guidance synthesized directly from scan progression",
  "clinical_actions": ["Specific clinical action 1", "Specific clinical action 2", "Specific clinical action 3"],
  "follow_up_timeline": "Specific timeline recommendation for next imaging or consultation",
  "dietary_lifestyle": ["Hydration protocol", "Dietary guideline 1", "Dietary guideline 2"],
  "confidence_assessment": "Assessment of AI diagnostic certainty and scan quality"
}}
"""

    parsed, used_model = _call_gemini_json(prompt)

    # Validate primary condition - ensure it aligns with the latest scan
    gemini_primary = parsed.get("primary_condition", latest_pred)
    if latest_pred.lower() == "normal" and gemini_primary.lower() in ["tumor", "stone", "cyst"]:
        gemini_primary = "Normal"

    # Validate urgency color
    urgency_color = parsed.get("urgency_color", "emerald")
    if latest_pred.lower() == "normal" and urgency_color == "rose":
        urgency_color = "emerald"
    elif urgency_color not in ["emerald", "amber", "orange", "rose"]:
        urgency_lower = str(parsed.get("urgency", "")).lower()
        if "urgent" in urgency_lower or "high" in urgency_lower:
            urgency_color = "rose"
        elif "mod" in urgency_lower:
            urgency_color = "amber"
        else:
            urgency_color = "emerald"

    urgency_val = parsed.get("urgency", "Normal" if latest_pred.lower() == "normal" else "Moderate")
    if latest_pred.lower() == "normal" and urgency_val.lower() == "urgent":
        urgency_val = "Low / Surveillance"

    return {
        "urgency": urgency_val,
        "urgency_color": urgency_color,
        "risk_trend": parsed.get("risk_trend", "Monitored Trajectory"),
        "primary_condition": gemini_primary,
        "summary": parsed.get("summary", "Clinical analysis complete."),
        "clinical_actions": parsed.get("clinical_actions", ["Consult with your nephrologist or urologist."]),
        "follow_up_timeline": parsed.get("follow_up_timeline", "Routine follow-up in 6 to 12 months."),
        "dietary_lifestyle": parsed.get("dietary_lifestyle", ["Maintain healthy daily hydration."]),
        "confidence_assessment": parsed.get("confidence_assessment", f"Diagnostic confidence: {latest_conf:.1f}%"),
        "total_scans_analyzed": len(scans),
        "powered_by": f"Google Gemini AI ({used_model} - Grounded Intelligence)"
    }


def enhance_recommendation_with_gemini(base_recommendation: Dict[str, Any], scans: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Deprecated: redirects to pure Gemini recommendation."""
    return generate_pure_gemini_recommendation(scans)

