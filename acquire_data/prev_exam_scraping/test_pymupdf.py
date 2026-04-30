import fitz  # PyMuPDF

def extract_text_pymupdf(pdf_path):
    text = []
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

if __name__ == "__main__":
    # pdf_file = "/home/kevin/git_repos/dat255-teaching-assistant/data/previous_exam/DAT255_V25_ordinær_løsningsforslag.pdf"
    pdf_file = "/home/kevin/git_repos/dat255-teaching-assistant/data/previous_exam/DAT255-eksamen-kont-2024.pdf"
    extracted_text = extract_text_pymupdf(pdf_file)
    
    print(extracted_text)
    
    # Optional: save to file
    with open("output_pymupdf3.txt", "w", encoding="utf-8") as f:
        f.write(extracted_text)
        
    import re
    _QUESTION_RE = re.compile(r"^Question\s+(\d+)\s*\((\d+)p\)\s*$", re.IGNORECASE)

    with open("output_pymupdf4.txt", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            stripped = line.strip()
            if "question" in stripped.lower() or "oppgave" in stripped.lower():
                matched = "✓ MATCH" if _QUESTION_RE.match(stripped) else "✗ NO MATCH"
                print(f"Line {i:4d} {matched}: {repr(stripped)}")
