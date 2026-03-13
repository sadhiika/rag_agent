import argparse
import time
from pathlib import Path
import arxiv


def download(query: str, max_results: int, output_dir: str):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    print(f"Searching: {query}")
    
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)
    
    count = 0
    for result in client.results(search):
        paper_id = result.entry_id.split("/")[-1]
        pdf_path = output / f"{paper_id}.pdf"
        
        if pdf_path.exists():
            continue
        
        try:
            result.download_pdf(dirpath=str(output), filename=f"{paper_id}.pdf")
            count += 1
            print(f"  [{count}] {paper_id}: {result.title[:50]}...")
            time.sleep(1)
        except Exception as e:
            print(f"  Failed: {e}")
    
    print(f"\nDownloaded {count} papers to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default="LLM evaluation bias fairness", help="Search query")
    parser.add_argument("--max", type=int, default=50, help="Max papers")
    parser.add_argument("--output", default="data/raw", help="Output dir")
    args = parser.parse_args()
    download(args.query, args.max, args.output)