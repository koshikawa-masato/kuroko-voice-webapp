"""
RAG (Retrieval Augmented Generation) for Kuroko Voice
Uses FAISS for vector similarity search with OpenAI embeddings
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

# Index storage
INDEX_DIR = Path.home() / ".kuroko_voice" / "rag_index"


@dataclass
class Document:
    """A document chunk with metadata"""
    content: str
    source: str  # file path or URL
    chunk_id: int
    metadata: Dict = None


class RAGEngine:
    def __init__(self, index_name: str = "default"):
        """Initialize RAG engine

        Args:
            index_name: Name for the index (allows multiple indexes)
        """
        self.index_name = index_name
        self.index_path = INDEX_DIR / index_name
        self.index = None
        self.documents: List[Document] = []
        self.openai_client = None
        self._available = None

    def _init_openai(self):
        if self.openai_client is None:
            from openai import OpenAI
            self.openai_client = OpenAI()
        return self.openai_client

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using OpenAI"""
        client = self._init_openai()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts in one API call"""
        client = self._init_openai()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)

        chunks = []
        start = 0
        while start < len(tokens):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = enc.decode(chunk_tokens)
            chunks.append(chunk_text)
            start = end - overlap

        return chunks

    @property
    def available(self) -> bool:
        if self._available is None:
            try:
                import faiss
                import tiktoken
                self._available = True
            except ImportError as e:
                print(f"RAG not available: {e}")
                self._available = False
        return self._available

    def index_directory(self, directory: str, extensions: List[str] = None):
        """Index all files in a directory

        Args:
            directory: Path to directory
            extensions: File extensions to include (default: .md only for efficiency)
        """
        if not self.available:
            return

        import faiss
        import numpy as np

        # Default to .md only (README, docs, etc.) for efficiency
        if extensions is None:
            extensions = ['.md']

        print(f"Indexing {directory}...")
        print(f"Extensions: {extensions}")

        self.documents = []
        all_chunks = []  # Collect all chunks first

        # Walk directory and collect chunks
        dir_path = Path(directory)
        for file_path in dir_path.rglob('*'):
            if file_path.is_file() and file_path.suffix in extensions:
                # Skip hidden and common ignore patterns
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(part in ['node_modules', 'venv', '__pycache__', 'dist', 'build']
                       for part in file_path.parts):
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if not content.strip():
                        continue

                    # Chunk the content
                    chunks = self._chunk_text(content)

                    for i, chunk in enumerate(chunks):
                        # Create document
                        doc = Document(
                            content=chunk,
                            source=str(file_path.relative_to(dir_path)),
                            chunk_id=i,
                            metadata={'full_path': str(file_path)}
                        )
                        self.documents.append(doc)
                        all_chunks.append(chunk)

                    print(f"  Found: {file_path.name} ({len(chunks)} chunks)")

                except Exception as e:
                    print(f"  Skip {file_path.name}: {e}")

        if not all_chunks:
            print("No documents found")
            return

        # Batch embed all chunks (much faster than one-by-one)
        print(f"\nEmbedding {len(all_chunks)} chunks in batches...")
        all_embeddings = []
        batch_size = 100  # OpenAI supports up to 2048

        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            embeddings = self._get_embeddings_batch(batch)
            all_embeddings.extend(embeddings)
            print(f"  Batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1} done")

        # Build FAISS index
        embeddings_array = np.array(all_embeddings, dtype='float32')
        dimension = embeddings_array.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_array)

        print(f"\nIndexed {len(self.documents)} chunks from {len(set(d.source for d in self.documents))} files")

        # Save index
        self._save_index()

    def _save_index(self):
        """Save index to disk"""
        import faiss

        self.index_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(self.index_path / "index.faiss"))

        # Save documents
        docs_data = [
            {
                'content': d.content,
                'source': d.source,
                'chunk_id': d.chunk_id,
                'metadata': d.metadata
            }
            for d in self.documents
        ]
        with open(self.index_path / "documents.json", 'w') as f:
            json.dump(docs_data, f)

        print(f"Index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load index from disk"""
        if not self.available:
            return False

        import faiss

        try:
            self.index = faiss.read_index(str(self.index_path / "index.faiss"))

            with open(self.index_path / "documents.json", 'r') as f:
                docs_data = json.load(f)

            self.documents = [
                Document(
                    content=d['content'],
                    source=d['source'],
                    chunk_id=d['chunk_id'],
                    metadata=d.get('metadata')
                )
                for d in docs_data
            ]

            print(f"Loaded index with {len(self.documents)} chunks")
            return True

        except FileNotFoundError:
            return False
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Document]:
        """Search for relevant documents

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of relevant documents
        """
        if not self.available or self.index is None:
            return []

        import numpy as np

        # Get query embedding
        query_embedding = np.array([self._get_embedding(query)], dtype='float32')

        # Search
        distances, indices = self.index.search(query_embedding, top_k)

        # Get documents
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])

        return results

    def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for a query

        Args:
            query: The user's question
            max_tokens: Maximum context tokens

        Returns:
            Formatted context string
        """
        docs = self.search(query, top_k=10)

        if not docs:
            return ""

        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        context_parts = []
        total_tokens = 0

        for doc in docs:
            doc_text = f"[Source: {doc.source}]\n{doc.content}\n"
            doc_tokens = len(enc.encode(doc_text))

            if total_tokens + doc_tokens > max_tokens:
                break

            context_parts.append(doc_text)
            total_tokens += doc_tokens

        return "\n---\n".join(context_parts)


def main():
    """Test RAG functionality"""
    import argparse

    parser = argparse.ArgumentParser(description="RAG Index Manager")
    parser.add_argument("command", choices=["index", "search", "info"])
    parser.add_argument("--dir", "-d", help="Directory to index")
    parser.add_argument("--query", "-q", help="Search query")
    parser.add_argument("--name", "-n", default="default", help="Index name")
    parser.add_argument("--ext", "-e", nargs="+", help="File extensions (default: .md)")
    args = parser.parse_args()

    rag = RAGEngine(index_name=args.name)

    if args.command == "index":
        if not args.dir:
            print("Error: --dir required for indexing")
            return
        extensions = args.ext if args.ext else None
        rag.index_directory(args.dir, extensions=extensions)

    elif args.command == "search":
        if not args.query:
            print("Error: --query required for search")
            return
        if not rag.load_index():
            print("No index found. Run 'index' first.")
            return
        results = rag.search(args.query)
        for doc in results:
            print(f"\n[{doc.source}]")
            print(doc.content[:200] + "..." if len(doc.content) > 200 else doc.content)

    elif args.command == "info":
        if rag.load_index():
            sources = set(d.source for d in rag.documents)
            print(f"Index: {args.name}")
            print(f"Documents: {len(rag.documents)} chunks from {len(sources)} files")
        else:
            print("No index found")


if __name__ == "__main__":
    main()
