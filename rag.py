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

    def _chunk_markdown_by_headings(self, text: str, source: str) -> List[Dict]:
        """Split markdown by h2/h3 headings for better context preservation.

        Returns list of dicts with 'content' and 'heading' keys.
        This preserves the section context for each chunk.
        """
        import re
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")

        # Split by h2 (##) and h3 (###) headings
        heading_pattern = re.compile(r'^(#{2,3})\s+(.+)$', re.MULTILINE)

        sections = []
        last_end = 0
        current_heading = "Introduction"

        for match in heading_pattern.finditer(text):
            # Get content before this heading
            if match.start() > last_end:
                content = text[last_end:match.start()].strip()
                if content:
                    sections.append({
                        'heading': current_heading,
                        'content': content
                    })

            current_heading = match.group(2).strip()
            last_end = match.end()

        # Don't forget the last section
        if last_end < len(text):
            content = text[last_end:].strip()
            if content:
                sections.append({
                    'heading': current_heading,
                    'content': content
                })

        # Further chunk large sections while preserving heading context
        chunks = []
        max_tokens = 500

        for section in sections:
            tokens = enc.encode(section['content'])
            if len(tokens) <= max_tokens:
                chunks.append({
                    'heading': section['heading'],
                    'content': section['content']
                })
            else:
                # Split large sections into smaller chunks
                sub_chunks = self._chunk_text(section['content'], chunk_size=max_tokens, overlap=50)
                for i, chunk in enumerate(sub_chunks):
                    heading = section['heading']
                    if len(sub_chunks) > 1:
                        heading = f"{section['heading']} (part {i + 1})"
                    chunks.append({
                        'heading': heading,
                        'content': chunk
                    })

        return chunks

    def _get_commit_messages(self, repo_path: str, max_commits: int = 50) -> List[Dict]:
        """Extract meaningful commit messages from a git repository.

        Only returns commits with substantial messages (>50 chars) as these
        are more likely to contain thoughtful decisions.
        """
        import subprocess

        try:
            result = subprocess.run(
                ['git', 'log', f'--max-count={max_commits * 2}',
                 '--format=%H|%an|%s|%b', '--no-merges'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                parts = line.split('|', 3)
                if len(parts) < 3:
                    continue

                sha, author, subject = parts[0], parts[1], parts[2]
                body = parts[3] if len(parts) > 3 else ""

                # Full message
                full_message = f"{subject}\n{body}".strip()

                # Only keep substantial commits (>50 chars = thoughtful messages)
                if len(full_message) > 50:
                    commits.append({
                        'sha': sha[:7],
                        'author': author,
                        'message': full_message
                    })

                if len(commits) >= max_commits:
                    break

            return commits

        except Exception as e:
            print(f"  Error reading commits: {e}")
            return []

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

    def index_directory(self, directory: str, extensions: List[str] = None,
                        max_documents: int = None, max_per_source: int = None):
        """Index all files in a directory

        Args:
            directory: Path to directory
            extensions: File extensions to include (default: .md only for efficiency)
            max_documents: Maximum total document chunks (None = unlimited)
            max_per_source: Maximum chunks per source/repo directory (None = unlimited)
        """
        if not self.available:
            return

        import faiss
        import numpy as np
        import re

        # Default to .md only (README, docs, etc.) for efficiency
        if extensions is None:
            extensions = ['.md']

        print(f"Indexing {directory}...")
        print(f"Extensions: {extensions}")
        if max_documents:
            print(f"Max documents: {max_documents}")
        if max_per_source:
            print(f"Max per source: {max_per_source}")

        self.documents = []
        all_chunks = []  # Collect all chunks first
        source_counts = {}  # Track chunks per source (repo)

        # Walk directory and collect chunks
        dir_path = Path(directory)
        for file_path in dir_path.rglob('*'):
            # Check total document limit
            if max_documents and len(all_chunks) >= max_documents:
                print(f"  Reached max documents limit ({max_documents})")
                break

            if file_path.is_file() and file_path.suffix in extensions:
                # Skip hidden and common ignore patterns
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(part in ['node_modules', 'venv', '__pycache__', 'dist', 'build']
                       for part in file_path.parts):
                    continue

                # Extract source (repo name) from path
                relative_path = str(file_path.relative_to(dir_path))
                source_name = relative_path.split('/')[0] if '/' in relative_path else 'root'

                # Check per-source limit
                if max_per_source and source_counts.get(source_name, 0) >= max_per_source:
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if not content.strip():
                        continue

                    # Chunk the content
                    chunks = self._chunk_text(content)

                    # Apply limits
                    chunks_to_add = chunks
                    if max_per_source:
                        remaining = max_per_source - source_counts.get(source_name, 0)
                        chunks_to_add = chunks[:remaining]
                    if max_documents:
                        remaining = max_documents - len(all_chunks)
                        chunks_to_add = chunks_to_add[:remaining]

                    for i, chunk in enumerate(chunks_to_add):
                        # Create document
                        doc = Document(
                            content=chunk,
                            source=str(file_path.relative_to(dir_path)),
                            chunk_id=i,
                            metadata={'full_path': str(file_path), 'repo': source_name}
                        )
                        self.documents.append(doc)
                        all_chunks.append(chunk)
                        source_counts[source_name] = source_counts.get(source_name, 0) + 1

                    if len(chunks_to_add) < len(chunks):
                        print(f"  Found: {file_path.name} ({len(chunks_to_add)}/{len(chunks)} chunks, limited)")
                    else:
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

    def index_directory_expert(self, directory: str, extensions: List[str] = None,
                                max_documents: int = None, max_per_source: int = None):
        """Index directory with Expert Mode features:
        - Heading-based chunking for better context
        - Commit message extraction
        - Enhanced metadata

        Args:
            directory: Path to directory
            extensions: File extensions to include (default: .md)
            max_documents: Maximum total document chunks
            max_per_source: Maximum chunks per source/repo directory
        """
        if not self.available:
            return

        import faiss
        import numpy as np

        if extensions is None:
            extensions = ['.md']

        print(f"[Expert Mode] Indexing {directory}...")
        print(f"Extensions: {extensions}")

        self.documents = []
        all_chunks = []
        source_counts = {}

        dir_path = Path(directory)

        # First pass: Index markdown files with heading-based chunking
        for file_path in dir_path.rglob('*'):
            if max_documents and len(all_chunks) >= max_documents:
                print(f"  Reached max documents limit ({max_documents})")
                break

            if file_path.is_file() and file_path.suffix in extensions:
                if any(part.startswith('.') for part in file_path.parts):
                    continue
                if any(part in ['node_modules', 'venv', '__pycache__', 'dist', 'build']
                       for part in file_path.parts):
                    continue

                relative_path = str(file_path.relative_to(dir_path))
                source_name = relative_path.split('/')[0] if '/' in relative_path else 'root'

                if max_per_source and source_counts.get(source_name, 0) >= max_per_source:
                    continue

                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    if not content.strip():
                        continue

                    # Use heading-based chunking for markdown
                    if file_path.suffix == '.md':
                        heading_chunks = self._chunk_markdown_by_headings(content, relative_path)
                    else:
                        # Fallback for non-md files
                        text_chunks = self._chunk_text(content)
                        heading_chunks = [{'heading': 'Content', 'content': c} for c in text_chunks]

                    chunks_to_add = heading_chunks
                    if max_per_source:
                        remaining = max_per_source - source_counts.get(source_name, 0)
                        chunks_to_add = chunks_to_add[:remaining]
                    if max_documents:
                        remaining = max_documents - len(all_chunks)
                        chunks_to_add = chunks_to_add[:remaining]

                    for i, chunk_data in enumerate(chunks_to_add):
                        # Enhanced content with heading context
                        enhanced_content = f"[{chunk_data['heading']}]\n{chunk_data['content']}"

                        doc = Document(
                            content=enhanced_content,
                            source=relative_path,
                            chunk_id=i,
                            metadata={
                                'full_path': str(file_path),
                                'repo': source_name,
                                'heading': chunk_data['heading'],
                                'type': 'documentation'
                            }
                        )
                        self.documents.append(doc)
                        all_chunks.append(enhanced_content)
                        source_counts[source_name] = source_counts.get(source_name, 0) + 1

                    print(f"  [Doc] {file_path.name} ({len(chunks_to_add)} sections)")

                except Exception as e:
                    print(f"  Skip {file_path.name}: {e}")

        # Second pass: Extract commit messages from repos
        print("\n[Expert Mode] Extracting commit messages...")
        for repo_dir in dir_path.iterdir():
            if not repo_dir.is_dir():
                continue
            if repo_dir.name.startswith('.'):
                continue

            git_dir = repo_dir / '.git'
            if not git_dir.exists():
                continue

            if max_documents and len(all_chunks) >= max_documents:
                break

            commits = self._get_commit_messages(str(repo_dir), max_commits=30)

            if commits:
                repo_name = repo_dir.name

                if max_per_source:
                    remaining = max_per_source - source_counts.get(repo_name, 0)
                    commits = commits[:remaining]
                if max_documents:
                    remaining = max_documents - len(all_chunks)
                    commits = commits[:remaining]

                for i, commit in enumerate(commits):
                    commit_content = f"[Commit {commit['sha']} by {commit['author']}]\n{commit['message']}"

                    doc = Document(
                        content=commit_content,
                        source=f"{repo_name}/commits",
                        chunk_id=i,
                        metadata={
                            'repo': repo_name,
                            'sha': commit['sha'],
                            'author': commit['author'],
                            'type': 'commit'
                        }
                    )
                    self.documents.append(doc)
                    all_chunks.append(commit_content)
                    source_counts[repo_name] = source_counts.get(repo_name, 0) + 1

                print(f"  [Commits] {repo_name}: {len(commits)} meaningful commits")

        if not all_chunks:
            print("No documents found")
            return

        # Batch embed all chunks
        print(f"\nEmbedding {len(all_chunks)} chunks in batches...")
        all_embeddings = []
        batch_size = 100

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

        # Count types
        doc_count = len([d for d in self.documents if d.metadata.get('type') == 'documentation'])
        commit_count = len([d for d in self.documents if d.metadata.get('type') == 'commit'])

        print(f"\n[Expert Mode] Indexed {len(self.documents)} total chunks:")
        print(f"  - {doc_count} documentation sections")
        print(f"  - {commit_count} commit messages")

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
