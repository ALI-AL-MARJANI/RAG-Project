import os
import json
import feedparser
import urllib.request

def fetch_recent_arxiv_ids(category="cs.LG", max_results=5):
    """
    Fetches the most recent arXiv paper IDs for a given category
    
    """
    url = f"http://export.arxiv.org/api/query?search_query=cat:{category}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    feed = feedparser.parse(url)

    ids = []
    for entry in feed.entries:
        paper_id = entry.id.split('/abs/')[-1]
        ids.append(paper_id)

    return ids


def download_arxiv_papers(output_dir="data/raw/arxiv", category="cs.LG", max_results=5):
    """
    Downloads recent arXiv papers in the specified category and saves them to output_dir
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"Fetching latest papers from arXiv ({category})…")
    paper_ids = fetch_recent_arxiv_ids(category, max_results)

    pdf_paths = []
    metadata = []

    for pid in paper_ids:
        base_id = pid.replace("v1", "")
        pdf_url = f"https://arxiv.org/pdf/{base_id}.pdf"
        pdf_path = os.path.join(output_dir, f"{base_id}.pdf")

        try:
            print(f"Downloading {pid} … ", end="")
            urllib.request.urlretrieve(pdf_url, pdf_path)
            print("OK")

            pdf_paths.append(pdf_path)
            metadata.append({
                "id": pid,
                "pdf_path": pdf_path,
                "url": f"https://arxiv.org/abs/{pid}",
            })
        except Exception as e:
            print(f"FAILED ({e})")

    # save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)

    return pdf_paths
