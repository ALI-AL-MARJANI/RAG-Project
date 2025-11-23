import os
from pathlib import Path
from pdfminer.high_level import extract_text



def parse_pdf(pdf_path: str):
    """
    Extract raw text from a PDF using pdfminer.
    
    Parameters
    ----------
    pdf_path : str
        Path to the PDF file.

    Returns
    -------
    str
        Extracted raw text.
    """
    try:
        text = extract_text(pdf_path)
        return text
    
    except Exception as e:
        print(f"Error while parsing {pdf_path}: {e}")
        return ""
    

def batch_parse_pdfs(input_dir="data/raw/arxiv", output_dir="data/processed/text"):
    """
    Parse all PDFs from input_dir and store their raw text in output_dir.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed_files = []

    for pdf in input_dir.glob("*.pdf"):
        print(f"Parsing {pdf.name}... ", end="")
        txt = parse_pdf(str(pdf))

        out_path = output_dir / f"{pdf.stem}.txt"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(txt)

        print("OK")
        parsed_files.append(str(out_path))

    print(f"\n{len(parsed_files)} files parsed.")
    return parsed_files
