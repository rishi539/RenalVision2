import os
import json
from datetime import datetime
from typing import List, Dict, Any

from sqlalchemy import create_engine, Column, Integer, String, Float, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from cnnClassifier import logger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DB_PATH = os.path.join(BASE_DIR, "model", "patient_history.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class UserModel(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False, index=True)
    email = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    name = Column(String, nullable=False)
    created_at = Column(String, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    scans = relationship("ScanModel", back_populates="user", cascade="all, delete-orphan")
    recommendation = relationship("RecommendationModel", back_populates="user", uselist=False, cascade="all, delete-orphan")


class ScanModel(Base):
    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    scan_date = Column(String, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    scan_number = Column(Integer, nullable=False)
    prediction = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    prob_normal = Column(Float, nullable=False)
    prob_cyst = Column(Float, nullable=False)
    prob_stone = Column(Float, nullable=False)
    prob_tumor = Column(Float, nullable=False)
    gradcam_b64 = Column(String, nullable=True)
    original_b64 = Column(String, nullable=True)

    user = relationship("UserModel", back_populates="scans")


class RecommendationModel(Base):
    """Stores exactly ONE single active recommendation record per patient (overwritten on new scans)."""
    __tablename__ = "recommendations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), unique=True, nullable=False, index=True)
    urgency = Column(String, nullable=False)
    urgency_color = Column(String, nullable=False)
    risk_trend = Column(String, nullable=False)
    primary_condition = Column(String, nullable=False)
    summary = Column(String, nullable=False)
    clinical_actions_json = Column(String, nullable=False)
    follow_up_timeline = Column(String, nullable=False)
    dietary_lifestyle_json = Column(String, nullable=False)
    confidence_assessment = Column(String, nullable=False)
    total_scans_analyzed = Column(Integer, nullable=False)
    updated_at = Column(String, default=lambda: datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    user = relationship("UserModel", back_populates="recommendation")


def init_db() -> None:
    """Initializes the database tables using SQLAlchemy ORM."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info(f"SQLAlchemy ORM initialized database at {DB_PATH}")


def register_user(username: str, email: str, password: str, name: str) -> Dict[str, Any]:
    """Registers a new patient account."""
    session = SessionLocal()
    try:
        existing = session.query(UserModel).filter((UserModel.username == username) | (UserModel.email == email)).first()
        if existing:
            return {"error": "Username or email already exists"}

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        user = UserModel(
            username=username,
            email=email,
            password=password,
            name=name,
            created_at=now
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at
        }
    finally:
        session.close()


def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticates a patient login."""
    session = SessionLocal()
    try:
        user = session.query(UserModel).filter(
            UserModel.username == username,
            UserModel.password == password
        ).first()

        if not user:
            return None

        return {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "name": user.name,
            "created_at": user.created_at
        }
    finally:
        session.close()


def add_scan_record(
    user_id: int,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float],
    gradcam_b64: str = None,
    original_b64: str = None
) -> Dict[str, Any]:
    """Saves a new scan record for the logged in patient."""
    session = SessionLocal()
    try:
        last_scan = session.query(ScanModel).filter(ScanModel.user_id == user_id).order_by(ScanModel.id.desc()).first()
        scan_number = (last_scan.scan_number + 1) if last_scan else 1
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        scan = ScanModel(
            user_id=user_id,
            scan_date=now,
            scan_number=scan_number,
            prediction=prediction,
            confidence=confidence,
            prob_normal=float(probabilities.get("Normal", 0.0)),
            prob_cyst=float(probabilities.get("Cyst", 0.0)),
            prob_stone=float(probabilities.get("Stone", 0.0)),
            prob_tumor=float(probabilities.get("Tumor", 0.0)),
            gradcam_b64=gradcam_b64,
            original_b64=original_b64
        )
        session.add(scan)
        session.commit()
        session.refresh(scan)
        return {
            "id": scan.id,
            "user_id": scan.user_id,
            "scan_date": scan.scan_date,
            "scan_number": scan.scan_number,
            "prediction": scan.prediction,
            "confidence": scan.confidence,
            "probabilities": {
                "Normal": scan.prob_normal,
                "Cyst": scan.prob_cyst,
                "Stone": scan.prob_stone,
                "Tumor": scan.prob_tumor
            }
        }
    finally:
        session.close()


def generate_longitudinal_recommendations(scans: List[Dict[str, Any]], trend_status: str) -> Dict[str, Any]:
    """
    Generates dynamic clinical recommendations by analyzing all historical scans of the patient.
    Considers trajectory (improving/worsening/stable), primary diagnosis over timeline,
    confidence levels, and scan frequency.
    """
    if not scans:
        return {
            "urgency": "Low",
            "urgency_color": "emerald",
            "risk_trend": "No Scans Logged",
            "primary_condition": "None",
            "summary": "No scan history is available. Upload your first renal CT scan to initiate diagnostic tracking.",
            "clinical_actions": ["Upload an abdominal/renal CT scan to establish a baseline health record."],
            "follow_up_timeline": "Schedule an initial renal ultrasound or CT scan upon physician advisory.",
            "dietary_lifestyle": [
                "Maintain baseline hydration (2 to 2.5 Liters of water daily).",
                "Limit sodium intake to under 2,000 mg/day.",
                "Maintain regular physical activity and monitor blood pressure."
            ],
            "confidence_assessment": "N/A",
            "total_scans_analyzed": 0
        }

    latest = scans[-1]
    latest_prediction = latest["prediction"]
    latest_confidence = latest["confidence"]
    num_scans = len(scans)

    # 1. Analyze condition occurrences & peak severity across ALL scans
    all_predictions = [s["prediction"] for s in scans]
    has_tumor = "Tumor" in all_predictions
    has_stone = "Stone" in all_predictions
    has_cyst = "Cyst" in all_predictions

    # Calculate average confidence across all scans
    avg_confidence = sum(s["confidence"] for s in scans) / num_scans

    # Determine risk progression trajectory across scans
    if num_scans == 1:
        risk_trend = "Initial Baseline Assessment"
    else:
        first_abnormal = 100.0 - scans[0]["prob_normal"]
        latest_abnormal = 100.0 - latest["prob_normal"]
        if latest_abnormal < first_abnormal - 5.0:
            risk_trend = f"Improving Trend (Abnormal risk down by {first_abnormal - latest_abnormal:.1f}%)"
        elif latest_abnormal > first_abnormal + 5.0:
            risk_trend = f"Progression Trend (Abnormal risk up by {latest_abnormal - first_abnormal:.1f}%)"
        else:
            risk_trend = "Stable Longitudinal Trajectory"

    # Primary condition (latest or highest priority across history)
    if latest_prediction != "Normal":
        primary_condition = latest_prediction
    elif has_tumor:
        primary_condition = "Tumor (Historical Record)"
    elif has_stone:
        primary_condition = "Stone (Historical Record)"
    elif has_cyst:
        primary_condition = "Cyst (Historical Record)"
    else:
        primary_condition = "Normal / Healthy"

    clinical_actions = []
    dietary_lifestyle = []
    follow_up = ""
    summary = ""

    if latest_prediction == "Tumor" or (has_tumor and latest_prediction != "Normal"):
        urgency = "Urgent"
        urgency_color = "rose"
        summary = f"Longitudinal analysis across {num_scans} scan(s) indicates potential renal lesion / tumor presence. Immediate clinical correlation is strongly advised."
        clinical_actions = [
            "Schedule an urgent evaluation with a Urologic Oncologist or Nephrologist.",
            "Request a high-resolution multiphasic contrast-enhanced CT or MRI for definitive staging.",
            "Bring full DICOM scan history and progression graph to your physician appointment."
        ]
        follow_up = "Immediate follow-up within 1 to 2 weeks."
        dietary_lifestyle = [
            "Avoid all non-prescribed NSAIDs (e.g. ibuprofen, naproxen) which may stress renal function.",
            "Strictly follow nephrologist-guided fluid and dietary protocols.",
            "Monitor blood pressure twice daily and log any hematuria (blood in urine) or flank pain."
        ]

    elif latest_prediction == "Stone" or (has_stone and latest_prediction != "Normal"):
        if "Progression" in risk_trend:
            urgency = "High"
            urgency_color = "orange"
        else:
            urgency = "Moderate"
            urgency_color = "amber"

        summary = f"Analysis across {num_scans} scan(s) shows active or recurring renal calculi (kidney stone). Progression trajectory is currently classified as '{risk_trend}'."
        clinical_actions = [
            "Consult a Urologist for non-invasive stone evaluation (KUB X-ray or non-contrast CT).",
            "Consider a 24-hour urine collection analysis to identify specific stone composition (calcium oxalate, uric acid, or struvite)."
        ]
        follow_up = "Follow-up scan in 4 to 8 weeks to monitor stone passage or size changes."
        dietary_lifestyle = [
            "Hydration Target: Drink at least 3.0 Liters of water daily to maintain dilute urine output.",
            "Dietary Sodium: Keep sodium under 1,500 - 2,000 mg/day to reduce urinary calcium excretion.",
            "Dietary Oxalate: Limit high-oxalate foods (spinach, rhubarb, almonds, dark chocolate) if oxalate stones are suspected.",
            "Citrate Boost: Add real lemon juice to water (natural citrate helps inhibit stone formation)."
        ]

    elif latest_prediction == "Cyst" or (has_cyst and latest_prediction != "Normal"):
        urgency = "Moderate"
        urgency_color = "amber"
        summary = f"Longitudinal history across {num_scans} scan(s) shows renal cyst findings. Cyst progression trajectory is '{risk_trend}'."
        clinical_actions = [
            "Schedule a routine urological consultation for Bosniak classification evaluation.",
            "Compare cyst dimensions with previous baseline scan (#1) to rule out rapid enlargement."
        ]
        follow_up = "Re-evaluation ultrasound or CT recommended in 3 to 6 months."
        dietary_lifestyle = [
            "Maintain healthy blood pressure (target < 120/80 mmHg) to minimize renal strain.",
            "Stay well-hydrated (2.5 Liters of water daily).",
            "Limit excessive caffeine and alcohol consumption."
        ]

    else:
        if num_scans > 1 and (has_stone or has_cyst or has_tumor):
            urgency = "Low (Recovered)"
            urgency_color = "emerald"
            summary = f"Latest scan (# {latest['scan_number']}) returned Normal/Healthy, showing positive recovery compared to earlier findings."
            clinical_actions = [
                "Continue routine preventive nephrology monitoring.",
                "Maintain periodic diagnostic scans as recommended by your primary care physician."
            ]
            follow_up = "Annual preventive renal scan or routine blood/urine health panel."
            dietary_lifestyle = [
                "Maintain optimal daily fluid intake (2 to 2.5 Liters).",
                "Balanced, low-sodium Mediterranean-style diet rich in vegetables and fiber.",
                "Avoid unnecessary overuse of OTC pain relievers."
            ]
        else:
            urgency = "Low"
            urgency_color = "emerald"
            summary = f"Longitudinal history across {num_scans} scan(s) shows consistent normal kidney architecture without detectable lesions."
            clinical_actions = [
                "No urgent clinical interventions required based on current scan history.",
                "Continue standard annual health checkups with your physician."
            ]
            follow_up = "Routine annual preventive health assessment."
            dietary_lifestyle = [
                "Maintain healthy hydration (2 to 2.5 Liters of water daily).",
                "Maintain balanced sodium and protein intake.",
                "Engage in regular cardiovascular exercise."
            ]

    if latest_confidence < 75.0:
        confidence_assessment = f"Latest scan confidence is moderate ({latest_confidence:.1f}%). Multi-scan correlation across all {num_scans} records was used for high-fidelity evaluation."
    else:
        confidence_assessment = f"High confidence rating ({latest_confidence:.1f}%) backed by longitudinal baseline across {num_scans} consecutive scan(s) (Avg confidence: {avg_confidence:.1f}%)."

    base_rec = {
        "urgency": urgency,
        "urgency_color": urgency_color,
        "risk_trend": risk_trend,
        "primary_condition": primary_condition,
        "summary": summary,
        "clinical_actions": clinical_actions,
        "follow_up_timeline": follow_up,
        "dietary_lifestyle": dietary_lifestyle,
        "confidence_assessment": confidence_assessment,
        "total_scans_analyzed": num_scans
    }

    # Enhance using Google Gemini Pro/Flash (if GEMINI_API_KEY is configured)
    from cnnClassifier.utils.gemini_recommender import enhance_recommendation_with_gemini
    return enhance_recommendation_with_gemini(base_rec, scans)


def save_or_update_user_recommendation(user_id: int, scans: List[Dict[str, Any]], trend_status: str) -> Dict[str, Any]:
    """
    Computes recommendation and overwrites (upserts) the single recommendation record for this patient in the DB.
    Guarantees exactly ONE recommendation record per patient.
    """
    rec = generate_longitudinal_recommendations(scans, trend_status)

    session = SessionLocal()
    try:
        existing = session.query(RecommendationModel).filter(RecommendationModel.user_id == user_id).first()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if existing:
            # Overwrite existing single recommendation record for this user
            existing.urgency = rec["urgency"]
            existing.urgency_color = rec["urgency_color"]
            existing.risk_trend = rec["risk_trend"]
            existing.primary_condition = rec["primary_condition"]
            existing.summary = rec["summary"]
            existing.clinical_actions_json = json.dumps(rec["clinical_actions"])
            existing.follow_up_timeline = rec["follow_up_timeline"]
            existing.dietary_lifestyle_json = json.dumps(rec["dietary_lifestyle"])
            existing.confidence_assessment = rec["confidence_assessment"]
            existing.total_scans_analyzed = rec["total_scans_analyzed"]
            existing.updated_at = now
        else:
            # Create a single recommendation record for this user
            new_rec = RecommendationModel(
                user_id=user_id,
                urgency=rec["urgency"],
                urgency_color=rec["urgency_color"],
                risk_trend=rec["risk_trend"],
                primary_condition=rec["primary_condition"],
                summary=rec["summary"],
                clinical_actions_json=json.dumps(rec["clinical_actions"]),
                follow_up_timeline=rec["follow_up_timeline"],
                dietary_lifestyle_json=json.dumps(rec["dietary_lifestyle"]),
                confidence_assessment=rec["confidence_assessment"],
                total_scans_analyzed=rec["total_scans_analyzed"],
                updated_at=now
            )
            session.add(new_rec)

        session.commit()
        return rec
    except Exception as e:
        session.rollback()
        logger.error(f"Error saving/updating recommendation for user {user_id}: {e}")
        return rec
    finally:
        session.close()





def get_user_history(user_id: int) -> Dict[str, Any]:
    """Fetches the logged in patient's profile details, scan timeline, and stored single recommendation record."""
    session = SessionLocal()
    try:
        user = session.query(UserModel).filter(UserModel.id == user_id).first()
        if not user:
            return None

        scans_query = session.query(ScanModel).filter(ScanModel.user_id == user_id).order_by(ScanModel.id.asc()).all()
        scans = [{
            "id": s.id,
            "user_id": s.user_id,
            "scan_date": s.scan_date,
            "scan_number": s.scan_number,
            "prediction": s.prediction,
            "confidence": s.confidence,
            "prob_normal": s.prob_normal,
            "prob_cyst": s.prob_cyst,
            "prob_stone": s.prob_stone,
            "prob_tumor": s.prob_tumor,
            "gradcam_b64": s.gradcam_b64,
            "original_b64": s.original_b64
        } for s in scans_query]

        trend_status = "No Scans Logged Yet"
        trend_color = "neutral"

        if len(scans) > 0:
            latest = scans[-1]
            if latest["prediction"] == "Normal":
                trend_status = "Condition Normal / Healthy"
                trend_color = "emerald"
            elif len(scans) >= 2:
                prev = scans[-2]
                prev_abnormal = 100.0 - prev["prob_normal"]
                latest_abnormal = 100.0 - latest["prob_normal"]
                if latest_abnormal < prev_abnormal:
                    trend_status = f"Improving ({latest['prediction']} severity decreasing)"
                    trend_color = "emerald"
                else:
                    trend_status = f"Monitored ({latest['prediction']} detected)"
                    trend_color = "amber" if latest["prediction"] in ["Cyst", "Stone"] else "rose"
            else:
                trend_status = f"Initial Diagnosis: {latest['prediction']}"
                trend_color = "amber" if latest["prediction"] in ["Cyst", "Stone"] else "rose"

        chart_series = {
            "labels": [f"Scan #{s['scan_number']} ({s['scan_date'].split(' ')[0]})" for s in scans],
            "normal": [s["prob_normal"] for s in scans],
            "cyst": [s["prob_cyst"] for s in scans],
            "stone": [s["prob_stone"] for s in scans],
            "tumor": [s["prob_tumor"] for s in scans],
        }

        # Fetch the patient's single stored recommendation record from database
        rec_model = session.query(RecommendationModel).filter(RecommendationModel.user_id == user_id).first()
        recommendations = None
        if rec_model:
            recommendations = {
                "urgency": rec_model.urgency,
                "urgency_color": rec_model.urgency_color,
                "risk_trend": rec_model.risk_trend,
                "primary_condition": rec_model.primary_condition,
                "summary": rec_model.summary,
                "clinical_actions": json.loads(rec_model.clinical_actions_json) if rec_model.clinical_actions_json else [],
                "follow_up_timeline": rec_model.follow_up_timeline,
                "dietary_lifestyle": json.loads(rec_model.dietary_lifestyle_json) if rec_model.dietary_lifestyle_json else [],
                "confidence_assessment": rec_model.confidence_assessment,
                "total_scans_analyzed": rec_model.total_scans_analyzed,
                "updated_at": rec_model.updated_at
            }

        return {
            "user": {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "name": user.name,
                "created_at": user.created_at
            },
            "scans": scans,
            "trend_status": trend_status,
            "trend_color": trend_color,
            "chart_series": chart_series,
            "recommendations": recommendations
        }
    finally:
        session.close()


def delete_scan_record(scan_id: int) -> bool:
    """Deletes a scan record by ID."""
    session = SessionLocal()
    try:
        scan = session.query(ScanModel).filter(ScanModel.id == scan_id).first()
        if not scan:
            return False
        session.delete(scan)
        session.commit()
        return True
    except Exception as e:
        logger.error(f"Error deleting scan: {e}")
        return False
    finally:
        session.close()


def trigger_gemini_recommendation_for_user(user_id: int) -> Dict[str, Any]:
    """Explicitly triggers Gemini AI recommendation analysis for a given user, updates single DB record, and returns."""
    session = SessionLocal()
    try:
        scans_query = session.query(ScanModel).filter(ScanModel.user_id == user_id).order_by(ScanModel.id.asc()).all()
        scans = [{
            "id": s.id, "user_id": s.user_id, "scan_date": s.scan_date, "scan_number": s.scan_number,
            "prediction": s.prediction, "confidence": s.confidence, "prob_normal": s.prob_normal,
            "prob_cyst": s.prob_cyst, "prob_stone": s.prob_stone, "prob_tumor": s.prob_tumor
        } for s in scans_query]

        from cnnClassifier.utils.gemini_recommender import generate_pure_gemini_recommendation
        rec = generate_pure_gemini_recommendation(scans)

        existing = session.query(RecommendationModel).filter(RecommendationModel.user_id == user_id).first()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if existing:
            existing.urgency = rec["urgency"]
            existing.urgency_color = rec["urgency_color"]
            existing.risk_trend = rec["risk_trend"]
            existing.primary_condition = rec["primary_condition"]
            existing.summary = rec["summary"]
            existing.clinical_actions_json = json.dumps(rec["clinical_actions"])
            existing.follow_up_timeline = rec["follow_up_timeline"]
            existing.dietary_lifestyle_json = json.dumps(rec["dietary_lifestyle"])
            existing.confidence_assessment = rec["confidence_assessment"]
            existing.total_scans_analyzed = rec["total_scans_analyzed"]
            existing.updated_at = now
        else:
            new_rec = RecommendationModel(
                user_id=user_id,
                urgency=rec["urgency"],
                urgency_color=rec["urgency_color"],
                risk_trend=rec["risk_trend"],
                primary_condition=rec["primary_condition"],
                summary=rec["summary"],
                clinical_actions_json=json.dumps(rec["clinical_actions"]),
                follow_up_timeline=rec["follow_up_timeline"],
                dietary_lifestyle_json=json.dumps(rec["dietary_lifestyle"]),
                confidence_assessment=rec["confidence_assessment"],
                total_scans_analyzed=rec["total_scans_analyzed"],
                updated_at=now
            )
            session.add(new_rec)

        session.commit()
        return rec
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to generate Gemini recommendation for user {user_id}: {e}")
        raise e
    finally:
        session.close()
