"""
Topic Diversity vs H-index Analysis
This script fetches author data from OpenAlex API and analyzes the relationship
between research topic diversity and H-index.
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from collections import Counter
import numpy as np
from datetime import datetime
import os

# Configuration
API_BASE = "https://api.openalex.org"
AUTHORS_LIMIT = 100
OUTPUT_DIR = "results"
EMAIL = "user@example.com"  # Required by OpenAlex API

os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_authors(limit=100):
    """Fetch a sample of authors from OpenAlex API."""
    print(f"Fetching {limit} authors from OpenAlex API...")
    
    authors = []
    url = f"{API_BASE}/authors"
    params = {
        "per_page": min(limit, 200),
        "mailto": EMAIL,
        "sort": "h_index:desc"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        for author in data.get("results", [])[:limit]:
            authors.append({
                "id": author.get("id"),
                "display_name": author.get("display_name"),
                "h_index": author.get("h_index"),
                "works_count": author.get("works_count")
            })
        
        print(f"✓ Successfully fetched {len(authors)} authors")
        return authors
    
    except requests.exceptions.RequestException as e:
        print(f"✗ Error fetching authors: {e}")
        return []

def fetch_author_works(author_id):
    """Fetch works for a specific author."""
    try:
        url = f"{API_BASE}/authors/{author_id}/works"
        params = {
            "per_page": 50,
            "mailto": EMAIL
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        works = []
        for work in data.get("results", []):
            concepts = [c.get("display_name") for c in work.get("concepts", [])]
            works.append({"concepts": concepts})
        
        return works
    
    except requests.exceptions.RequestException as e:
        print(f"  Warning: Could not fetch works for author {author_id}: {e}")
        return []

def compute_topic_diversity(works):
    """Compute topic diversity score."""
    if not works:
        return 0, 0
    
    all_concepts = []
    for work in works:
        all_concepts.extend(work.get("concepts", []))
    
    unique_concepts = len(set(all_concepts))
    
    if all_concepts:
        concept_counts = Counter(all_concepts)
        total = len(all_concepts)
        probabilities = [count / total for count in concept_counts.values()]
        entropy = -sum(p * np.log(p) for p in probabilities if p > 0)
    else:
        entropy = 0
    
    return unique_concepts, entropy

def analyze_authors(authors):
    """Analyze all authors."""
    print("\nAnalyzing authors...")
    analysis_results = []
    
    for idx, author in enumerate(authors, 1):
        print(f"  [${idx}/${len(authors)}] Processing {author['display_name']}...", end=" ")
        
        if author["h_index"] is None:
            print("⊘ (no h_index)")
            continue
        
        works = fetch_author_works(author["id"])\n        
        if not works:
            print("⊘ (no works)")
            continue
        
        unique_concepts, entropy = compute_topic_diversity(works)
        
        analysis_results.append({
            "author_name": author["display_name"],
            "h_index": author["h_index"],
            "works_count": len(works),
            "unique_concepts": unique_concepts,
            "entropy": entropy,
            "diversity_score": unique_concepts
        })
        
        print(f"✓ (h_index={author['h_index']}, concepts={unique_concepts})")
    
    print(f"\n✓ Analysis complete: {len(analysis_results)} authors processed")
    return analysis_results

def visualize_results(df, output_path="results/scatter_plot.png"):
    """Create scatter plot."""
    plt.figure(figsize=(12, 8))
    
    if len(df) < 2:
        print("Not enough data to create visualization")
        return
    
    correlation, p_value = pearsonr(df["diversity_score"], df["h_index"])
    corr_text = f"Pearson r = {correlation:.3f}, p-value = {p_value:.4f}"
    
    plt.scatter(df["diversity_score"], df["h_index"], alpha=0.6, s=100, color="steelblue")
    
    z = np.polyfit(df["diversity_score"], df["h_index"], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df["diversity_score"].min(), df["diversity_score"].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label="Trend line")
    
    plt.xlabel("Topic Diversity Score (Unique Concepts)", fontsize=12, fontweight="bold")
    plt.ylabel("H-index", fontsize=12, fontweight="bold")
    plt.title(f"Research Topic Diversity vs H-index\n{corr_text}", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✓ Plot saved to {output_path}")
    plt.close()

def main():
    """Main execution function."""
    print("="*80)
    print("TOPIC DIVERSITY vs H-INDEX ANALYSIS")
    print("="*80)
    
authors = fetch_authors(AUTHORS_LIMIT)
    
    if not authors:
        print("✗ Failed to fetch authors. Exiting.")
        return
    
    analysis_results = analyze_authors(authors)
    
    if not analysis_results:
        print("✗ No analysis results. Exiting.")
        return
    
    df = pd.DataFrame(analysis_results)
    df = df.dropna()
    
    print(f"\nDataset: {len(df)} authors with complete data")
    
    df.to_csv(f"{OUTPUT_DIR}/analysis_data.csv", index=False)
    print(f"✓ Dataset saved to {OUTPUT_DIR}/analysis_data.csv")
    
    visualize_results(df)
    
    print("\n" + "="*80)
    print("✓ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()