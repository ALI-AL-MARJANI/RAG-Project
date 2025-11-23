import os
import json
from pathlib import Path
import requests


def download_arxiv_papers(arxiv_ids, output_dir="data/raw/arxiv"):
    """
    Download arXiv papers (PDF) based on a list of arXiv IDs.
    
    Parameters
    ----------
    arxiv_ids : list[str]
        List of arXiv identifiers, e.g. ["2402.06782", "2310.12345"]
    output_dir : str
        Directory where PDFs and metadata will be stored
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []

    for arxiv_id in arxiv_ids:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        pdf_path = output_dir / f"{arxiv_id}.pdf"

        print(f"Downloading {arxiv_id}... ", end="")

        try:
            response = requests.get(pdf_url, timeout=20)

            if response.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(response.content)

                print("OK")
                metadata.append(
                    {"arxiv_id": arxiv_id, "pdf_path": str(pdf_path), "url": pdf_url}
                )
            else:
                print(f"FAILED (status {response.status_code})")

        except Exception as e:
            print(f"ERROR: {e}")

    # Save metadata
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"\nDownloaded {len(metadata)} papers. Metadata saved in {meta_path}")
    return metadata
