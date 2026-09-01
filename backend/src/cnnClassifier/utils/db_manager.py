import os
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

    user = relationship("UserModel", back_populates="scans")


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
    gradcam_b64: str = None
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
            gradcam_b64=gradcam_b64
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


def get_user_history(user_id: int) -> Dict[str, Any]:
    """Fetches the logged in patient's profile details, scan timeline, and longitudinal trend series."""
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
            "gradcam_b64": s.gradcam_b64
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
            "chart_series": chart_series
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
