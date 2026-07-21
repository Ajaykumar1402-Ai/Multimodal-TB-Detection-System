from fpdf import FPDF
import os
import datetime


def generate_pdf_report(patient_name: str, record_data: dict, record_id: int, output_dir: str = "./reports") -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pdf = FPDF()
    pdf.set_margins(12, 12, 12)
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    now = datetime.datetime.now()

    # ─── HEADER BAND ────────────────────────────────────────────────────────────
    pdf.set_fill_color(0, 48, 107)          # deep blue
    pdf.rect(0, 0, 210, 14, 'F')
    pdf.set_font("Arial", 'B', 11)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(8, 3)
    pdf.cell(0, 8, "DIAGNOSTIC REPORT", ln=0)

    pdf.set_font("Arial", 'B', 13)
    pdf.set_xy(0, 3)
    pdf.cell(210, 8, "TB-VISION AI", align='C', ln=0)

    pdf.set_font("Arial", '', 8)
    pdf.set_xy(150, 3)
    pdf.cell(52, 8, "Multimodal TB Detection System", align='R', ln=0)

    # -- PATIENT INFO BOX -------------------------------------------------------
    pdf.ln(6)
    pdf.set_text_color(0, 0, 0)

    # Left patient photo placeholder
    pdf.set_fill_color(220, 230, 240)
    pdf.rect(12, 18, 28, 28, 'F')
    pdf.set_font("Arial", 'I', 7)
    pdf.set_text_color(90, 90, 90)
    pdf.set_xy(12, 30)
    pdf.cell(28, 5, "Patient Photo", align='C')

    # Center: Ref Doctor
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 9)
    pdf.set_xy(44, 19)
    safe_doctor = str(record_data.get('doctor_email', 'N/A'))
    pdf.cell(80, 6, f"REF. DOCTOR :  {safe_doctor}", ln=1)

    # Right table: DOB, AGE/SEX, DRAWN, RECEIVED, REPORTED
    info_x = 130
    pdf.set_font("Arial", '', 8)
    
    # Calculate age fallback if age missing but dob exists
    safe_age = record_data.get('age', 'N/A')
    
    rows = [
        ("DATE OF BIRTH", str(record_data.get('date_of_birth') or 'N/A')),
        ("AGE",  f"{safe_age} Years"),
        ("DRAWN",    now.strftime('%d/%m/%Y %H:%M:%S')),
        ("RECEIVED", now.strftime('%d/%m/%Y %H:%M:%S')),
        ("REPORTED", now.strftime('%d/%m/%Y %H:%M:%S')),
    ]
    for i, (label, val) in enumerate(rows):
        pdf.set_xy(info_x, 18 + i * 5.5)
        pdf.set_font("Arial", 'B', 8)
        pdf.cell(24, 5, label, border=0)
        pdf.set_font("Arial", '', 8)
        pdf.cell(0, 5, f": {val}", ln=0)

    # Patient name strip (Making it bolder and slightly larger to guarantee visibility)
    safe_patient = str(patient_name).strip() if patient_name else "UNKNOWN PATIENT"
    pdf.set_fill_color(240, 244, 248)
    pdf.set_xy(44, 32)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(80, 8, f"PATIENT: {safe_patient.title()}", fill=True, ln=1)
    
    # Clinical ID below name
    pdf.set_xy(44, 40)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(80, 5, f"CLINICAL ID: #{record_data.get('patient_id', 'N/A')}", ln=1)

    pdf.ln(18)

    # ─── COLUMN HEADERS ─────────────────────────────────────────────────────────
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(12)
    pdf.cell(90, 7, "Test Report Status     Final", fill=True, border=1)
    pdf.cell(40, 7, "Results", fill=True, border=1, align='C')
    pdf.cell(40, 7, "Reference Interval", fill=True, border=1, align='C')
    pdf.cell(20, 7, "Units", fill=True, border=1, align='C')
    pdf.ln()

    # ─── SECTION TITLE ──────────────────────────────────────────────────────────
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", 'B', 9)
    pdf.set_x(12)
    pdf.cell(190, 7, "   AI-BASED TUBERCULOSIS DIAGNOSTIC ANALYSIS (TBFERON-AI)", fill=True, border=1, ln=1)

    # ─── Helper to draw a test row ───────────────────────────────────────────────
    def draw_row(name, method, result_val, reference="", unit="", highlight=False, bold_result=False):
        pdf.set_x(12)
        row_y = pdf.get_y()

        # Zebra stripe
        if highlight:
            pdf.set_fill_color(255, 245, 245)
        else:
            pdf.set_fill_color(255, 255, 255)

        # Test name
        pdf.set_font("Arial", 'B', 8)
        pdf.set_text_color(0, 0, 0)
        pdf.multi_cell(90, 5, name, border='LR', fill=highlight)
        name_h = pdf.get_y() - row_y

        # Method below name
        pdf.set_xy(12, row_y + 5)
        pdf.set_font("Arial", 'I', 6.5)
        pdf.set_text_color(100, 100, 100)
        pdf.cell(90, 4, f"  METHOD : {method}", border='LR', fill=highlight, ln=0)

        row_end_y = pdf.get_y() + 4

        # Result
        pdf.set_xy(102, row_y)
        if bold_result:
            pdf.set_font("Arial", 'B', 9)
        else:
            pdf.set_font("Arial", '', 8)
        # Color result by risk
        if "HIGH" in str(result_val).upper():
            pdf.set_text_color(180, 0, 0)
        elif "MEDIUM" in str(result_val).upper() or "MOD" in str(result_val).upper():
            pdf.set_text_color(180, 100, 0)
        elif "LOW" in str(result_val).upper() or "NEG" in str(result_val).upper():
            pdf.set_text_color(0, 120, 0)
        else:
            pdf.set_text_color(0, 0, 0)
        pdf.cell(40, row_end_y - row_y, str(result_val), border='LR', align='C', fill=highlight)

        # Reference
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", '', 8)
        pdf.cell(40, row_end_y - row_y, reference, border='LR', align='C', fill=highlight)

        # Unit
        pdf.cell(20, row_end_y - row_y, unit, border='LR', align='C', fill=highlight)
        pdf.ln()
        pdf.set_text_color(0, 0, 0)

    # ─── TEST ROWS ───────────────────────────────────────────────────────────────
    final_prob = record_data.get("final_probability", 0)
    cnn_prob   = record_data.get("cnn_probability", 0)
    clin_prob  = record_data.get("clinical_probability", 0)
    risk       = record_data.get("risk_level", "Low")

    draw_row("SPECIMEN SOURCE", "DIGITAL CHEST X-RAY + CLINICAL DATA",
             "MULTIMODAL INPUT", "", "")

    draw_row("CNN X-RAY PROBABILITY",
             "DEEP LEARNING - MOBILENETV2",
             f"{cnn_prob * 100:.1f}%",
             "< 42.0%", "%",
             highlight=(cnn_prob >= 0.42),
             bold_result=True)

    draw_row("CLINICAL PARAMETER SCORE",
             "LOGISTIC REGRESSION ML MODEL",
             f"{clin_prob * 100:.1f}%",
             "< 42.0%", "%",
             highlight=(clin_prob >= 0.42),
             bold_result=True)

    ci = record_data.get("confidence_interval")
    if ci:
        result_label = f"{final_prob * 100:.1f}%  (95% CI: {ci[0]*100:.0f}%-{ci[1]*100:.0f}%)  {risk}"
    else:
        result_label = f"{final_prob * 100:.1f}%  {risk}"
        
    draw_row("FINAL FUSED TB PROBABILITY",
             "WEIGHTED MULTIMODAL FUSION (60% IMAGE + 40% CLINICAL)",
             result_label,
             "< 42.0%", "%",
             highlight=(final_prob >= 0.42),
             bold_result=True)

    # Clinical inputs
    fever       = "Yes" if record_data.get("fever") else "No"
    wt_loss     = "Yes" if record_data.get("weight_loss") else "No"
    night_sw    = "Yes" if record_data.get("night_sweats") else "No"
    cough_wks   = record_data.get("cough_duration_weeks", 0)
    draw_row("COUGH DURATION", "CLINICAL PARAMETER",
             f"{cough_wks} Weeks", "> 2 Weeks indicates risk", "Weeks")
    draw_row("FEVER / WEIGHT LOSS / NIGHT SWEATS", "CLINICAL PARAMETER",
             f"{fever} / {wt_loss} / {night_sw}", "", "")

    # Lab Data Rows
    sputum_val = "Positive" if record_data.get("sputum_test") == 1 else ("Negative" if record_data.get("sputum_test") == 2 else "Not Done")
    genexpert_val = "Positive" if record_data.get("genexpert_test") == 1 else ("Negative" if record_data.get("genexpert_test") == 2 else "Not Done")
    
    draw_row("SPUTUM SMEAR MICROSCOPY", "LAB DATA / ACID-FAST BACILLI",
             sputum_val, "NEGATIVE", "", highlight=(sputum_val == "Positive"))
    draw_row("GENEXPERT MTB/RIF", "LAB DATA / NUCLEIC ACID AMPLIFICATION",
             genexpert_val, "NEGATIVE", "", highlight=(genexpert_val == "Positive"))

    # INTERPRETATION row
    interpretation = "POSITIVE" if final_prob >= 0.42 else "NEGATIVE"
    draw_row("INTERPRETATION",
             "AI MULTIMODAL FUSION ANALYSIS",
             interpretation,
             "NEGATIVE", "",
             highlight=(final_prob >= 0.42),
             bold_result=True)

    pdf.ln(3)

    # ─── AFFECTED REGIONS TABLE ───────────────────────────────────────────────
    regions = record_data.get("affected_regions", [])
    if regions:
        pdf.set_fill_color(0, 48, 107)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Arial", 'B', 8)
        pdf.set_x(12)
        pdf.cell(190, 6, "   LUNG REGION CLASSIFICATION (MedSAM AI Segmentation)", fill=True, border=0, ln=1)

        pdf.set_text_color(0, 0, 0)
        pdf.set_fill_color(230, 230, 230)
        pdf.set_font("Arial", 'B', 8)
        pdf.set_x(12)
        pdf.cell(110, 6, "Anatomical Region", fill=True, border=1)
        pdf.cell(40, 6, "Severity", fill=True, border=1, align='C')
        pdf.cell(40, 6, "Status", fill=True, border=1, align='C')
        pdf.ln()

        severity_status = {"HIGH": "Critically Affected", "MEDIUM": "Moderately Affected", "LOW": "Mildly Affected"}
        severity_colors = {"HIGH": (180, 0, 0), "MEDIUM": (180, 100, 0), "LOW": (0, 130, 0)}
        for region in regions:
            sev = region.get("severity", "LOW")
            r, g, b = severity_colors.get(sev, (0, 0, 0))
            pdf.set_x(12)
            pdf.set_font("Arial", '', 8)
            pdf.set_text_color(0, 0, 0)
            pdf.cell(110, 6, f"  {region.get('region', '')}", border=1)
            pdf.set_font("Arial", 'B', 8)
            pdf.set_text_color(r, g, b)
            pdf.cell(40, 6, sev, border=1, align='C')
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", '', 8)
            pdf.cell(40, 6, severity_status.get(sev, ""), border=1, align='C')
            pdf.ln()

    pdf.ln(4)

    # ─── INTERPRETATION NOTES ────────────────────────────────────────────────────
    pdf.set_fill_color(245, 245, 245)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_text_color(0, 0, 0)
    pdf.set_x(12)
    pdf.cell(190, 6, "Interpretation(s)", fill=True, border='B', ln=1)

    pdf.set_font("Arial", '', 7.5)
    rec_text = record_data.get("recommendations", "")
    interp = (
        "The AI-based TB detection system uses a multimodal approach combining Deep Learning analysis of the Chest X-Ray image "
        "with clinical parameter scoring (cough duration, fever, weight loss, night sweats) to arrive at a fused probability score.\n\n"
        f"AI Recommendation: {rec_text}\n\n"
        "DISCLAIMER: This report is generated by an AI system for clinical decision support only. It is not intended to replace "
        "professional medical diagnosis. A qualified physician should evaluate these results in the context of the patient's full "
        "clinical picture. Confirmatory tests (sputum culture, GeneXpert, TST) are recommended for definitive TB diagnosis."
    )
    pdf.set_x(12)
    pdf.multi_cell(190, 5, interp)

    pdf.ln(8)

    # ─── SIGNATURE ───────────────────────────────────────────────────────────────
    pdf.set_font("Arial", 'I', 8)
    pdf.set_x(12)
    pdf.cell(95, 5, "_________________________", ln=0)
    pdf.cell(95, 5, "_________________________", ln=1)
    pdf.set_font("Arial", 'B', 8)
    pdf.set_x(12)
    pdf.cell(95, 5, "Authorized Signatory", ln=0)
    pdf.cell(95, 5, "Reviewing Physician", ln=1)
    pdf.set_font("Arial", '', 7)
    pdf.set_x(12)
    pdf.cell(95, 4, "TB-Vision AI Diagnostic System", ln=0)
    pdf.cell(95, 4, record_data.get('doctor_email', ''), ln=1)

    pdf.ln(3)
    pdf.set_font("Arial", '', 7)
    pdf.set_x(12)
    pdf.cell(0, 4, f"Page 1 Of 1", align='R', ln=1)

    # ─── FOOTER BAND ─────────────────────────────────────────────────────────────
    pdf.set_fill_color(0, 48, 107)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", '', 7)
    footer_y = 282
    pdf.set_xy(0, footer_y)
    pdf.cell(210, 5, "", fill=True, ln=1)
    pdf.set_xy(12, footer_y)
    pdf.cell(0, 5,
             "PERFORMED BY: TB-Vision AI  |  Report generated automatically  |  System Version 1.0",
             fill=True)

    # ─── SAVE ────────────────────────────────────────────────────────────────────
    safe_name = patient_name.replace(' ', '_').replace('/', '_')
    # Stable naming convention using record_id
    file_path = os.path.join(output_dir, f"report_{safe_name}_{record_id}.pdf")
    pdf.output(file_path)
    return file_path
