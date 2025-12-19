"""
Company Embedding Module - Generate embeddings for company details using Jina Embeddings v4.

This module processes company information and generates vector embeddings that can be
used for semantic search, similarity computation, and clustering tasks.

Jina Embeddings v4 Features:
- Multimodal support (text + images)
- Multilingual capability (including Chinese)
- Task-specific LoRA adapters for optimal performance
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv
from tqdm import tqdm


# ============================================================================
# Configuration Constants
# ============================================================================

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
DEFAULT_MODEL = "jina-embeddings-v4"
DEFAULT_DIMENSIONS = 1024
DEFAULT_TASK = "retrieval.passage"
DEFAULT_BATCH_SIZE = 32
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class CompanyRecord:
    """Raw company record from CSV."""
    company_id: str
    company_name: str
    location: str
    company_details: str


@dataclass
class CompanyEmbedding:
    """Embedding result for a single company."""
    company_id: str
    company_name: str
    embedding: List[float] = field(default_factory=list)
    token_count: int = 0
    error: str = ""


# ============================================================================
# Company Embedder Class
# ============================================================================

class CompanyEmbedder:
    """Generate embeddings for company details using Jina Embeddings v4 API.
    
    Features:
    - Batch processing with configurable batch size
    - Automatic retry with exponential backoff
    - Progress tracking with tqdm
    - Checkpoint support for resumable processing
    
    Example:
        embedder = CompanyEmbedder(api_key="your_key")
        embeddings = embedder.embed_companies(companies)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        dimensions: int = DEFAULT_DIMENSIONS,
        task: str = DEFAULT_TASK,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Initialize the embedder.
        
        Args:
            api_key: Jina AI API key
            model: Model name (default: jina-embeddings-v4)
            dimensions: Embedding dimensions (default: 1024, max: 2048)
            task: Task type for LoRA adapter (default: retrieval.passage)
            batch_size: Number of texts per API request (default: 32)
        """
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.task = task
        self.batch_size = batch_size
        self._total_tokens = 0
    
    @property
    def total_tokens_used(self) -> int:
        """Total tokens used in this session."""
        return self._total_tokens
    
    def _make_request(
        self, 
        texts: List[str], 
        retries: int = MAX_RETRIES
    ) -> Tuple[List[List[float]], int]:
        """Make API request with retry logic.
        
        Args:
            texts: List of texts to embed
            retries: Number of retries remaining
            
        Returns:
            Tuple of (embeddings list, total tokens used)
            
        Raises:
            Exception: If all retries are exhausted
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "input": texts,
            "model": self.model,
            "dimensions": self.dimensions,
            "task": self.task,
        }
        
        try:
            response = requests.post(
                JINA_API_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            
            data = response.json()
            embeddings = [item["embedding"] for item in data["data"]]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            
            return embeddings, tokens
            
        except requests.exceptions.RequestException as e:
            if retries > 0:
                delay = RETRY_DELAY * (MAX_RETRIES - retries + 1)
                time.sleep(delay)
                return self._make_request(texts, retries - 1)
            raise Exception(f"API request failed after {MAX_RETRIES} retries: {str(e)}")
    
    def embed_single(self, text: str) -> Tuple[List[float], int]:
        """Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Tuple of (embedding vector, tokens used)
        """
        embeddings, tokens = self._make_request([text])
        self._total_tokens += tokens
        return embeddings[0], tokens
    
    def embed_batch(
        self, 
        texts: List[str],
        show_progress: bool = False,
    ) -> List[Tuple[List[float], int]]:
        """Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            show_progress: Whether to show progress bar
            
        Returns:
            List of (embedding, tokens) tuples for each text
        """
        results = []
        
        # Process in batches
        batches = [
            texts[i:i + self.batch_size] 
            for i in range(0, len(texts), self.batch_size)
        ]
        
        iterator = batches
        if show_progress:
            iterator = tqdm(batches, desc=f"Embedding ({self.model})")
        
        for batch in iterator:
            try:
                embeddings, tokens = self._make_request(batch)
                self._total_tokens += tokens
                
                # Distribute tokens evenly for per-item tracking
                tokens_per_item = tokens // len(batch)
                for emb in embeddings:
                    results.append((emb, tokens_per_item))
                    
            except Exception as e:
                # On batch failure, add empty results with error
                for _ in batch:
                    results.append(([], 0))
        
        return results
    
    def embed_company(self, company: CompanyRecord) -> CompanyEmbedding:
        """Generate embedding for a single company.
        
        Args:
            company: Company record to process
            
        Returns:
            CompanyEmbedding with the result
        """
        text = self._prepare_company_text(company)
        
        try:
            embedding, tokens = self.embed_single(text)
            return CompanyEmbedding(
                company_id=company.company_id,
                company_name=company.company_name,
                embedding=embedding,
                token_count=tokens,
            )
        except Exception as e:
            return CompanyEmbedding(
                company_id=company.company_id,
                company_name=company.company_name,
                error=str(e),
            )
    
    def embed_companies(
        self,
        companies: List[CompanyRecord],
        show_progress: bool = True,
        checkpoint_path: Optional[Path] = None,
    ) -> List[CompanyEmbedding]:
        """Generate embeddings for multiple companies.
        
        Args:
            companies: List of company records to process
            show_progress: Whether to show progress bar
            checkpoint_path: Path to save/load checkpoints for resumable processing
            
        Returns:
            List of CompanyEmbedding results
        """
        # Load checkpoint if exists
        processed_ids = set()
        results = []
        
        if checkpoint_path and checkpoint_path.exists():
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)
                for item in checkpoint_data:
                    results.append(CompanyEmbedding(**item))
                    processed_ids.add(item["company_id"])
            if show_progress:
                print(f"Loaded {len(results)} embeddings from checkpoint")
        
        # Filter out already processed companies
        remaining = [c for c in companies if c.company_id not in processed_ids]
        
        if not remaining:
            return results
        
        # Prepare texts
        texts = [self._prepare_company_text(c) for c in remaining]
        
        # Process in batches
        batches = [
            (remaining[i:i + self.batch_size], texts[i:i + self.batch_size])
            for i in range(0, len(remaining), self.batch_size)
        ]
        
        iterator = batches
        if show_progress:
            processed_batches = len(results) // self.batch_size if results else 0
            total_batches = processed_batches + len(batches)
            iterator = tqdm(
                batches, 
                desc=f"Embedding ({self.model})",
                initial=processed_batches,
                total=total_batches,
            )
        
        for company_batch, text_batch in iterator:
            try:
                embeddings, tokens = self._make_request(text_batch)
                self._total_tokens += tokens
                tokens_per_item = tokens // len(text_batch)
                
                for company, embedding in zip(company_batch, embeddings):
                    results.append(CompanyEmbedding(
                        company_id=company.company_id,
                        company_name=company.company_name,
                        embedding=embedding,
                        token_count=tokens_per_item,
                    ))
                    
            except Exception as e:
                # On batch failure, process individually
                for company, text in zip(company_batch, text_batch):
                    try:
                        embedding, tokens = self._make_request([text])
                        self._total_tokens += tokens
                        results.append(CompanyEmbedding(
                            company_id=company.company_id,
                            company_name=company.company_name,
                            embedding=embedding[0],
                            token_count=tokens,
                        ))
                    except Exception as inner_e:
                        results.append(CompanyEmbedding(
                            company_id=company.company_id,
                            company_name=company.company_name,
                            error=str(inner_e),
                        ))
            
            # Save checkpoint after each batch
            if checkpoint_path:
                self._save_checkpoint(results, checkpoint_path)
        
        # Sort by company_id for consistent output
        results.sort(key=lambda x: x.company_id)
        return results
    
    def _prepare_company_text(self, company: CompanyRecord) -> str:
        """Prepare text for embedding from company record.
        
        Combines company name, location, and details for richer semantic representation.
        """
        return f"{company.company_name}\n{company.location}\n{company.company_details}"
    
    def _save_checkpoint(self, results: List[CompanyEmbedding], path: Path) -> None:
        """Save checkpoint for resumable processing."""
        data = [asdict(r) for r in results]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)


# ============================================================================
# Utility Functions
# ============================================================================

def load_companies_from_csv(csv_path: Path) -> List[CompanyRecord]:
    """Load company records from CSV file."""
    import csv
    
    companies = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            companies.append(CompanyRecord(
                company_id=row.get("company_id", ""),
                company_name=row.get("company_name", ""),
                location=row.get("location", ""),
                company_details=row.get("company_details", ""),
            ))
    return companies


def save_embeddings_csv(
    results: List[CompanyEmbedding], 
    output_path: Path,
    include_vector: bool = True,
) -> None:
    """Save embedding results to CSV.
    
    Args:
        results: List of CompanyEmbedding objects
        output_path: Path to output CSV file
        include_vector: Whether to include the embedding vector (as JSON string)
    """
    import csv
    
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        
        if include_vector:
            writer.writerow(["company_id", "company_name", "embedding", "token_count", "error"])
            for result in results:
                writer.writerow([
                    result.company_id,
                    result.company_name,
                    json.dumps(result.embedding) if result.embedding else "",
                    result.token_count,
                    result.error,
                ])
        else:
            writer.writerow(["company_id", "company_name", "token_count", "error"])
            for result in results:
                writer.writerow([
                    result.company_id,
                    result.company_name,
                    result.token_count,
                    result.error,
                ])


def save_embeddings_json(results: List[CompanyEmbedding], output_path: Path) -> None:
    """Save embedding results to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_embeddings_npy(results: List[CompanyEmbedding], output_path: Path) -> None:
    """Save embeddings as NumPy array for efficient computation.
    
    Also saves a mapping file for company_id to index.
    """
    import numpy as np
    
    # Filter out results with errors
    valid_results = [r for r in results if r.embedding]
    
    if not valid_results:
        print("Warning: No valid embeddings to save")
        return
    
    # Create numpy array
    embeddings = np.array([r.embedding for r in valid_results], dtype=np.float32)
    np.save(output_path, embeddings)
    
    # Save mapping
    mapping = {r.company_id: i for i, r in enumerate(valid_results)}
    mapping_path = output_path.with_suffix(".mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def load_embeddings_npy(npy_path: Path) -> Tuple[Any, Dict[str, int]]:
    """Load embeddings from NumPy file with mapping.
    
    Returns:
        Tuple of (numpy array, company_id to index mapping)
    """
    import numpy as np
    
    embeddings = np.load(npy_path)
    mapping_path = npy_path.with_suffix(".mapping.json")
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    
    return embeddings, mapping


def load_jina_api_key() -> str:
    """Load Jina API key from environment."""
    load_dotenv()
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "JINA_API_KEY is required. Please set it in your .env file or environment."
        )
    return api_key


def print_embedding_summary(results: List[CompanyEmbedding]) -> None:
    """Print summary statistics of embedding results."""
    print("\n" + "=" * 60)
    print("EMBEDDING SUMMARY")
    print("=" * 60)
    
    total = len(results)
    successful = sum(1 for r in results if r.embedding)
    failed = sum(1 for r in results if r.error)
    total_tokens = sum(r.token_count for r in results)
    
    print(f"Total companies processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total tokens used: {total_tokens:,}")
    
    if successful > 0:
        first_valid = next(r for r in results if r.embedding)
        print(f"Embedding dimensions: {len(first_valid.embedding)}")
    
    if failed > 0:
        print("\nFailed companies:")
        for r in results:
            if r.error:
                print(f"  - {r.company_id}: {r.error[:100]}")


# ============================================================================
# CLI Entry Point (for standalone usage)
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate embeddings for company details using Jina Embeddings v4"
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to input CSV file with company data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output_embeddings"),
        help="Directory for output files",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=DEFAULT_DIMENSIONS,
        choices=[128, 256, 512, 1024, 2048],
        help=f"Embedding dimensions (default: {DEFAULT_DIMENSIONS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of companies to process (for testing)",
    )
    
    args = parser.parse_args()
    
    # Load API key
    api_key = load_jina_api_key()
    
    # Load companies
    print(f"Loading companies from {args.input_csv}...")
    companies = load_companies_from_csv(args.input_csv)
    print(f"Loaded {len(companies)} companies")
    
    if args.limit:
        companies = companies[:args.limit]
        print(f"Limited to {len(companies)} companies")
    
    # Initialize embedder
    embedder = CompanyEmbedder(
        api_key=api_key,
        dimensions=args.dimensions,
    )
    
    # Process
    print(f"\nGenerating embeddings with {embedder.model}...")
    print(f"Dimensions: {embedder.dimensions}")
    
    results = embedder.embed_companies(companies, show_progress=True)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = args.output_dir / "company_embeddings.csv"
    save_embeddings_csv(results, csv_path)
    print(f"Saved CSV: {csv_path}")
    
    json_path = args.output_dir / "company_embeddings.json"
    save_embeddings_json(results, json_path)
    print(f"Saved JSON: {json_path}")
    
    npy_path = args.output_dir / "company_embeddings.npy"
    save_embeddings_npy(results, npy_path)
    print(f"Saved NPY: {npy_path}")
    
    # Print summary
    print_embedding_summary(results)
