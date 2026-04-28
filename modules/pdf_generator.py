import os
import tempfile
import datetime
from fpdf import FPDF
import re

def strip_emojis_and_markdown(text: str) -> str:
    """Removes emojis and basic markdown so fpdf doesn't crash on standard fonts."""
    # Remove specific emojis we know we use
    text = text.replace("⚠️  ", "").replace("✅  ", "")
    text = text.replace("🔴 ", "").replace("🟡 ", "").replace("🟢 ", "")
    
    # Remove markdown asterisks, backticks, hashes, and angle brackets
    text = re.sub(r'[*`#>]', '', text)
    return text.strip()

def generate_pdf_report(code: str, language: str, syntax_status: str, vulnerable: str, vuln_type: str, severity: str, risk_score: str, confidence: str, affected_lines: str, reasoning: str) -> str:
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Helvetica", size=18, style='B')
    pdf.cell(0, 12, txt="VulnDetect Security Audit Report", ln=True, align="C")
    
    # Date
    pdf.set_font("Helvetica", size=10)
    date_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    pdf.cell(0, 8, txt=f"Generated on: {date_str}", ln=True, align="C")
    pdf.ln(8)
    
    # Summary Section
    pdf.set_font("Helvetica", size=14, style='B')
    pdf.cell(0, 10, txt="1. Scan Results", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", size=11)
    
    # Clean the inputs
    c_vuln = strip_emojis_and_markdown(vulnerable)
    c_type = strip_emojis_and_markdown(vuln_type)
    c_sev = strip_emojis_and_markdown(severity)
    c_syntax = strip_emojis_and_markdown(syntax_status)
    
    # Helper to print rows
    def print_row(label, value):
        pdf.set_font("Helvetica", size=11, style='B')
        pdf.cell(45, 8, txt=label)
        pdf.set_font("Helvetica", size=11)
        # Handle long values by replacing unsupported chars
        safe_val = value.encode('latin-1', 'replace').decode('latin-1')
        pdf.multi_cell(0, 8, txt=safe_val)
        
    print_row("Verdict:", c_vuln)
    print_row("Syntax Status:", c_syntax)
    print_row("Vulnerability Type:", c_type)
    print_row("Severity:", c_sev)
    print_row("Risk Score:", risk_score)
    print_row("Confidence:", confidence)
    print_row("Affected Lines:", affected_lines)
    
    pdf.ln(8)
    
    # AI Reasoning Section
    pdf.set_font("Helvetica", size=14, style='B')
    pdf.cell(0, 10, txt="2. AI Security Audit Reasoning", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Helvetica", size=11)
    clean_reasoning = strip_emojis_and_markdown(reasoning)
    safe_reasoning = clean_reasoning.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, txt=safe_reasoning)
    
    pdf.ln(10)
    
    # Code Snippet Section
    pdf.set_font("Helvetica", size=14, style='B')
    pdf.cell(0, 10, txt=f"3. Analyzed Code ({language})", ln=True)
    pdf.ln(2)
    
    pdf.set_font("Courier", size=9)
    # Background for code block (fpdf doesn't do multi_cell backgrounds easily, so we just print text)
    safe_code = code.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 5, txt=safe_code, border=1)
    
    # Save to temp file
    fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="VulnDetect_Report_")
    os.close(fd)
    
    pdf.output(temp_path)
    return temp_path
