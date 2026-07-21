from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func, extract
from ..db import database, models
import datetime

router = APIRouter()

@router.get("/dashboard")
def get_dashboard_stats(db: Session = Depends(database.get_db)):
    # 1. Basic counts
    total_screenings = db.query(models.DiagnosisRecord).count()
    high_risk = db.query(models.DiagnosisRecord).filter(models.DiagnosisRecord.risk_level == "High").count()
    cleared = db.query(models.DiagnosisRecord).filter(models.DiagnosisRecord.risk_level == "Low").count()
    
    # 2. Monthly Trend Data for 2026
    # Group by month for the current year
    current_year = 2026
    monthly_stats = db.query(
        extract('month', models.DiagnosisRecord.date).label('month'),
        func.count(models.DiagnosisRecord.id).label('screenings'),
        func.sum(models.DiagnosisRecord.final_tb_probability >= 0.4).label('positives')
    ).filter(extract('year', models.DiagnosisRecord.date) == current_year)\
     .group_by('month').all()
    
    # Format for Recharts
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    chart_data = []
    
    # Initialize with all months up to current (June in seeding)
    for i in range(1, 7):
        month_stat = next((m for m in monthly_stats if int(m.month) == i), None)
        chart_data.append({
            "name": month_names[i-1],
            "screenings": month_stat.screenings if month_stat else 0,
            "positives": int(month_stat.positives) if month_stat and month_stat.positives else 0
        })
        
    return {
        "total": total_screenings,
        "highRisk": high_risk,
        "resolved": cleared,
        "chartData": chart_data
    }
