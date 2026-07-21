from fpdf import FPDF
import os
import datetime

def generate_pdf_report(patient_name: str, record_data: dict, output_dir: str = "./reports") -> str:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Multimodal TB Detection Report", ln=True, align='C')
    pdf.ln(10)
    
    # Patient Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Patient Name: {patient_name}", ln=True)
    pdf.cell(200, 10, f"Patient ID: #{record_data.get('patient_id', 'N/A')}", ln=True)
    pdf.cell(200, 10, f"Patient Age: {record_data.get('age', 'N/A')} Years", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(5)
    
    # Clinical Data
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Clinical Data Inputs:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"- Cough Duration: {record_data.get('cough_duration_weeks')} weeks", ln=True)
    pdf.cell(200, 10, f"- Fever present: {'Yes' if record_data.get('fever') else 'No'}", ln=True)
    pdf.cell(200, 10, f"- Weight loss: {'Yes' if record_data.get('weight_loss') else 'No'}", ln=True)
    pdf.cell(200, 10, f"- Night sweats: {'Yes' if record_data.get('night_sweats') else 'No'}", ln=True)
    pdf.ln(5)
    
    # ML Results
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "AI Analysis Results:", ln=True)
    pdf.set_font("Arial", '', 12)
    pdf.cell(200, 10, f"- X-Ray CNN Probability: {record_data.get('cnn_probability', 0)*100:.2f}%", ln=True)
    pdf.cell(200, 10, f"- Clinical Model Probability: {record_data.get('clinical_probability', 0)*100:.2f}%", ln=True)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"- Final Fused TB Probability: {record_data.get('final_tb_probability', 0)*100:.2f}%", ln=True)
    
    # Risk & Recs
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Risk Level: {record_data.get('risk_level')}", ln=True)
    pdf.set_font("Arial", 'I', 11)
    pdf.multi_cell(0, 10, f"Recommendations: {record_data.get('recommendations')}")
    
    file_path = os.path.join(output_dir, f"report_{patient_name.replace(' ', '_')}_{int(datetime.datetime.now().timestamp())}.pdf")
    pdf.output(file_path)
    return file_path
