"""Async vector store using SimpleVecDB for persistent embedding storage.

Provides a unified async interface for:
- Embedding generation (via sentence-transformers)
- Persistent vector storage (via SimpleVecDB/SQLite)
- Semantic similarity search
- Hybrid search (BM25 + vector)

Designed for MCP server use with full async/await support.
"""

from __future__ import annotations

import asyncio
import hashlib
import sys
import threading
import types
from collections import OrderedDict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from loguru import logger


# =============================================================================
# Stub langchain_core and llama_index BEFORE any simplevecdb import
# This is required because simplevecdb's __init__.py imports integrations
# that depend on these packages, but we don't need those integrations.
# =============================================================================
def _stub_optional_deps() -> None:
    """Stub langchain_core and llama_index to allow simplevecdb import."""
    if "simplevecdb" in sys.modules:
        return  # Already imported, too late to stub

    def _stub_class(name: str) -> type:
        return type(name, (object,), {"__init__": lambda self, *a, **kw: None})

    # Stub langchain_core
    if "langchain_core" not in sys.modules:
        lc = types.ModuleType("langchain_core")
        lc.vectorstores = types.ModuleType("langchain_core.vectorstores")  # type: ignore[attr-defined]
        lc.embeddings = types.ModuleType("langchain_core.embeddings")  # type: ignore[attr-defined]
        lc.documents = types.ModuleType("langchain_core.documents")  # type: ignore[attr-defined]
        lc.vectorstores.VectorStore = _stub_class("VectorStore")  # type: ignore[attr-defined]
        lc.embeddings.Embeddings = _stub_class("Embeddings")  # type: ignore[attr-defined]
        lc.documents.Document = _stub_class("Document")  # type: ignore[attr-defined]
        sys.modules["langchain_core"] = lc
        sys.modules["langchain_core.vectorstores"] = lc.vectorstores  # type: ignore[assignment]
        sys.modules["langchain_core.embeddings"] = lc.embeddings  # type: ignore[assignment]
        sys.modules["langchain_core.documents"] = lc.documents  # type: ignore[assignment]

    # Stub llama_index
    if "llama_index" not in sys.modules:
        li = types.ModuleType("llama_index")
        li.core = types.ModuleType("llama_index.core")  # type: ignore[attr-defined]
        li.core.vector_stores = types.ModuleType("llama_index.core.vector_stores")  # type: ignore[attr-defined]
        li.core.schema = types.ModuleType("llama_index.core.schema")  # type: ignore[attr-defined]
        li.core.vector_stores.types = types.ModuleType("llama_index.core.vector_stores.types")  # type: ignore[attr-defined]
        li.core.vector_stores.VectorStoreQuery = _stub_class("VectorStoreQuery")  # type: ignore[attr-defined]
        li.core.vector_stores.VectorStoreQueryResult = _stub_class("VectorStoreQueryResult")  # type: ignore[attr-defined]
        li.core.schema.BaseNode = _stub_class("BaseNode")  # type: ignore[attr-defined]
        li.core.schema.TextNode = _stub_class("TextNode")  # type: ignore[attr-defined]
        li.core.vector_stores.types.BasePydanticVectorStore = _stub_class("BasePydanticVectorStore")  # type: ignore[attr-defined]
        li.core.vector_stores.types.MetadataFilters = _stub_class("MetadataFilters")  # type: ignore[attr-defined]
        li.core.vector_stores.types.VectorStoreQueryMode = _stub_class("VectorStoreQueryMode")  # type: ignore[attr-defined]
        sys.modules["llama_index"] = li
        sys.modules["llama_index.core"] = li.core  # type: ignore[assignment]
        sys.modules["llama_index.core.vector_stores"] = li.core.vector_stores  # type: ignore[assignment]
        sys.modules["llama_index.core.schema"] = li.core.schema  # type: ignore[assignment]
        sys.modules["llama_index.core.vector_stores.types"] = li.core.vector_stores.types  # type: ignore[assignment]


# Run stubbing immediately at module load
_stub_optional_deps()

# Now we can safely import simplevecdb types for TYPE_CHECKING
# Note: pyright may show import errors since simplevecdb isn't fully typed
if TYPE_CHECKING:
    from simplevecdb.core import VectorCollection, VectorDB  # type: ignore[import-not-found]

# Lazy imports for optional dependencies
_sentence_transformer_model = None
_sentence_transformer_lock = threading.Lock()


class DistanceMetric(str, Enum):
    """Distance metrics for similarity search."""

    COSINE = "cosine"
    L2 = "l2"
    DOT = "dot"


@dataclass
class SearchResult:
    """Result from a similarity search.

    Attributes:
        text: Original text content.
        score: Similarity/distance score (interpretation depends on metric).
        metadata: Associated metadata dictionary.
        doc_id: Database document ID.

    """

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: int | None = None


@dataclass
class VectorStoreConfig:
    """Configuration for VectorStore.

    Attributes:
        db_path: Path to SQLite database file.
        collection_name: Name of the vector collection.
        embedding_model: Sentence-transformer model name.
        embedding_dim: Embedding dimension (auto-detected if None).
        distance_metric: Distance metric for similarity search.
        use_quantization: Whether to use int8 quantization for storage.
        cache_size: LRU cache size for embeddings.
        cache_max_memory_mb: Maximum cache memory in megabytes.
        max_workers: Thread pool size for async operations.

    """

    db_path: str | Path = "ghostdm.db"
    collection_name: str = "thoughts"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int | None = None  # Auto-detect
    distance_metric: DistanceMetric = DistanceMetric.COSINE
    use_quantization: bool = False
    cache_size: int = 1000
    cache_max_memory_mb: float = 50.0
    max_workers: int = 4


class EmbeddingCache:
    """Thread-safe LRU cache for embedding vectors.

    Reduces redundant embedding computation for repeated texts.
    """

    def __init__(self, max_size: int = 1000, max_memory_mb: float = 50.0) -> None:
        """Initialize embedding cache.

        Args:
            max_size: Maximum number of cached embeddings.
            max_memory_mb: Maximum memory usage in megabytes.

        """
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self._current_memory = 0
        self.hits = 0
        self.misses = 0

    def _key(self, text: str, model: str) -> str:
        """Generate cache key from text and model name."""
        content = f"{model}:{text}"
        return hashlib.md5(content.encode(), usedforsecurity=False).hexdigest()

    def get(self, text: str, model: str) -> np.ndarray | None:
        """Get cached embedding (thread-safe).

        Args:
            text: Original text.
            model: Model name used for embedding.

        Returns:
            Cached embedding array or None if not found.

        """
        key = self._key(text, model)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key].copy()
            self.misses += 1
            return None

    def put(self, text: str, model: str, embedding: np.ndarray) -> None:
        """Cache an embedding (thread-safe).

        Args:
            text: Original text.
            model: Model name used for embedding.
            embedding: Embedding vector to cache.

        """
        key = self._key(text, model)
        emb_size = embedding.nbytes

        with self._lock:
            # Evict until we have space
            while (
                self._current_memory + emb_size > self.max_memory_bytes
                or len(self._cache) >= self.max_size
            ) and self._cache:
                _, old_emb = self._cache.popitem(last=False)
                self._current_memory -= old_emb.nbytes

            # Update existing key
            if key in self._cache:
                old = self._cache.pop(key)
                self._current_memory -= old.nbytes

            self._cache[key] = embedding.copy()
            self._current_memory += emb_size

    def clear(self) -> None:
        """Clear all cached embeddings."""
        with self._lock:
            self._cache.clear()
            self._current_memory = 0
            self.hits = 0
            self.misses = 0

    @property
    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total > 0 else 0.0,
                "memory_mb": self._current_memory / (1024 * 1024),
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
            }


def _get_embedding_model(model_name: str) -> Any:
    """Lazy-load the sentence transformer model (thread-safe singleton).

    Args:
        model_name: HuggingFace model name.

    Returns:
        SentenceTransformer model instance.

    """
    global _sentence_transformer_model

    with _sentence_transformer_lock:
        if (
            _sentence_transformer_model is None
            or getattr(_sentence_transformer_model, "_model_name", None) != model_name
        ):
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {model_name}")
                _sentence_transformer_model = SentenceTransformer(model_name)
                _sentence_transformer_model._model_name = model_name  # type: ignore
                logger.info(
                    f"Loaded embedding model with dim="
                    f"{_sentence_transformer_model.get_sentence_embedding_dimension()}"
                )
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers required for embedding generation. "
                    "Install with: pip install sentence-transformers"
                ) from e

        return _sentence_transformer_model


def _import_simplevecdb() -> tuple[Any, Any, Any, Any]:
    """Import simplevecdb (stubbing already done at module load).

    Returns:
        Tuple of (VectorDB, VectorCollection, DistanceStrategy, Quantization).

    """
    # Stubbing is done at module level by _stub_optional_deps()
    from simplevecdb.core import VectorCollection, VectorDB  # type: ignore[import-not-found]
    from simplevecdb.types import DistanceStrategy, Quantization  # type: ignore[import-not-found]

    return VectorDB, VectorCollection, DistanceStrategy, Quantization


class AsyncVectorStore:
    """Async vector store with embedding generation and persistent storage.

    Combines:
    - sentence-transformers for embedding generation
    - SimpleVecDB for SQLite-backed vector storage
    - LRU cache for embedding reuse
    - Full async/await API for MCP server integration

    Example:
        >>> store = AsyncVectorStore()
        >>> await store.add_texts(["hello world", "goodbye world"])
        >>> results = await store.search("greeting", k=1)
        >>> print(results[0].text)  # "hello world"

    """

    def __init__(self, config: VectorStoreConfig | None = None) -> None:
        """Initialize async vector store.

        Args:
            config: Store configuration. Uses defaults if None.

        """
        self.config = config or VectorStoreConfig()
        self._cache = EmbeddingCache(
            max_size=self.config.cache_size,
            max_memory_mb=self.config.cache_max_memory_mb,
        )
        self._db: VectorDB | None = None
        self._collection: VectorCollection | None = None
        self._embedding_dim: int | None = self.config.embedding_dim
        self._executor: ThreadPoolExecutor | None = None
        self._init_lock = asyncio.Lock()
        self._closed = False

    async def _ensure_initialized(self) -> None:
        """Lazy-initialize database and executor (async-safe)."""
        if self._db is not None:
            return

        async with self._init_lock:
            if self._db is not None:
                return

            # Create executor for CPU-bound operations
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

            try:
                VectorDB, _, DistanceStrategy, Quantization = _import_simplevecdb()
            except ImportError as e:
                raise ImportError(
                    "simplevecdb required for vector storage. Install with: pip install simplevecdb"
                ) from e

            # Map distance metric
            dist_map = {
                DistanceMetric.COSINE: DistanceStrategy.COSINE,
                DistanceMetric.L2: DistanceStrategy.L2,
                DistanceMetric.DOT: DistanceStrategy.COSINE,
            }

            quant = Quantization.INT8 if self.config.use_quantization else Quantization.FLOAT

            # Use sync VectorDB wrapped with async executor
            db = VectorDB(
                path=str(self.config.db_path),
                distance_strategy=dist_map.get(
                    self.config.distance_metric, DistanceStrategy.COSINE
                ),
                quantization=quant,
            )
            self._db = db
            self._collection = db.collection(self.config.collection_name)
            logger.debug(f"Initialized VectorDB at {self.config.db_path}")

    def _embed_texts_sync(self, texts: list[str]) -> np.ndarray:
        """Synchronous embedding generation (runs in executor).

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).

        """
        model = _get_embedding_model(self.config.embedding_model)

        # Check cache for each text
        embeddings: list[np.ndarray | None] = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text, self.config.embedding_model)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached texts
        if uncached_texts:
            new_embeddings = model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            # Cache and fill in results
            for i, (idx, text) in enumerate(zip(uncached_indices, uncached_texts, strict=False)):
                emb = new_embeddings[i]
                self._cache.put(text, self.config.embedding_model, emb)
                embeddings[idx] = emb

        # Stack into single array
        result = np.stack([e for e in embeddings if e is not None])

        # Update embedding dim if not set
        if self._embedding_dim is None:
            self._embedding_dim = result.shape[1]

        return result

    async def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts (async, runs in executor).

        Args:
            texts: List of texts to embed.

        Returns:
            Array of shape (len(texts), embedding_dim).

        """
        await self._ensure_initialized()
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, lambda: self._embed_texts_sync(texts))

    async def _embed_single(self, text: str) -> np.ndarray:
        """Embed a single text string (async).

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.

        """
        result = await self._embed_texts([text])
        return result[0]

    async def _run_sync(self, fn: Any) -> Any:
        """Run a synchronous function in the executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn)

    async def add_texts(
        self,
        texts: Sequence[str],
        metadatas: Sequence[dict[str, Any]] | None = None,
        ids: Sequence[int] | None = None,
    ) -> list[int]:
        """Add texts with optional metadata to the store.

        Args:
            texts: Text content to store.
            metadatas: Optional metadata dicts (one per text).
            ids: Optional document IDs for upsert behavior.

        Returns:
            List of document IDs.

        """
        await self._ensure_initialized()

        if not texts:
            return []

        # Generate embeddings
        embeddings = await self._embed_texts(list(texts))

        # Add to collection with pre-computed embeddings (sync call in executor)
        assert self._collection is not None

        def _add() -> list[int]:
            result = self._collection.add_texts(  # type: ignore
                texts=list(texts),
                metadatas=list(metadatas) if metadatas else None,
                embeddings=embeddings.tolist(),
                ids=list(ids) if ids else None,
            )
            return list(result) if result else []

        result: list[int] = await self._run_sync(_add)
        return result

    async def add_thought(
        self,
        thought: str,
        session_id: str,
        step: int,
        strategy: str | None = None,
        score: float | None = None,
        **extra_metadata: Any,
    ) -> int:
        """Add a reasoning thought to the store.

        Convenience method for storing reasoning steps.

        Args:
            thought: The thought content.
            session_id: Reasoning session identifier.
            step: Step number in the reasoning chain.
            strategy: Optional reasoning strategy used.
            score: Optional survival/confidence score.
            **extra_metadata: Additional metadata fields.

        Returns:
            Document ID.

        """
        metadata = {
            "session_id": session_id,
            "step": step,
            "type": "thought",
            **extra_metadata,
        }
        if strategy:
            metadata["strategy"] = strategy
        if score is not None:
            metadata["score"] = score

        ids = await self.add_texts([thought], [metadata])
        return ids[0]

    async def search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar texts.

        Args:
            query: Query text.
            k: Number of results to return.
            filter: Optional metadata filter.

        Returns:
            List of SearchResult objects, sorted by similarity.

        """
        await self._ensure_initialized()
        assert self._collection is not None

        # Embed query
        query_embedding = await self._embed_single(query)

        # Search (sync call in executor)
        def _search() -> list[tuple[Any, float]]:
            return self._collection.similarity_search(  # type: ignore
                query=query_embedding.tolist(),
                k=k,
                filter=filter,
            )

        results = await self._run_sync(_search)

        # Convert to SearchResult
        return [
            SearchResult(
                text=doc.page_content,
                score=score,
                metadata=doc.metadata,
                doc_id=doc.metadata.get("id"),
            )
            for doc, score in results
        ]

    async def search_by_session(
        self,
        query: str,
        session_id: str,
        k: int = 5,
    ) -> list[SearchResult]:
        """Search within a specific reasoning session.

        Args:
            query: Query text.
            session_id: Session to search within.
            k: Number of results.

        Returns:
            List of SearchResult objects from the session.

        """
        return await self.search(query, k=k, filter={"session_id": session_id})

    async def hybrid_search(
        self,
        query: str,
        k: int = 5,
        filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Hybrid search combining BM25 and vector similarity.

        Uses Reciprocal Rank Fusion to merge keyword and semantic results.

        Args:
            query: Query text (used for both keyword and vector search).
            k: Number of results.
            filter: Optional metadata filter.

        Returns:
            List of SearchResult objects.

        """
        await self._ensure_initialized()
        assert self._collection is not None

        # Embed query for vector component
        query_embedding = await self._embed_single(query)

        def _hybrid() -> list[tuple[Any, float]]:
            return self._collection.hybrid_search(  # type: ignore
                query=query,
                k=k,
                filter=filter,
                query_vector=query_embedding.tolist(),
            )

        results = await self._run_sync(_hybrid)

        return [
            SearchResult(
                text=doc.page_content,
                score=score,
                metadata=doc.metadata,
                doc_id=doc.metadata.get("id"),
            )
            for doc, score in results
        ]

    async def find_similar_thoughts(
        self,
        thought: str,
        session_id: str | None = None,
        k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """Find thoughts similar to a given thought.

        Useful for detecting redundancy or finding supporting evidence.

        Args:
            thought: Thought to find similar items for.
            session_id: Optional session to limit search.
            k: Maximum results to return.
            threshold: Optional similarity threshold (lower = more similar for cosine).

        Returns:
            List of similar thoughts.

        """
        filter_dict = {"session_id": session_id} if session_id else None
        results = await self.search(thought, k=k, filter=filter_dict)

        if threshold is not None:
            # For cosine distance: 0 = identical, 2 = opposite
            results = [r for r in results if r.score <= threshold]

        return results

    async def get_session_thoughts(
        self,
        session_id: str,
        limit: int = 50,
    ) -> list[SearchResult]:
        """Get all thoughts from a session.

        Args:
            session_id: Session identifier.
            limit: Maximum thoughts to return.

        Returns:
            List of thoughts from the session, sorted by step.

        """
        await self._ensure_initialized()
        assert self._collection is not None

        # Use similarity search with dummy vector to get all from session
        dim = self.embedding_dim

        def _get_all() -> list[tuple[Any, float]]:
            return self._collection.similarity_search(  # type: ignore
                query=[0.0] * dim,
                k=limit,
                filter={"session_id": session_id},
            )

        results = await self._run_sync(_get_all)

        # Convert and sort by step
        search_results = [
            SearchResult(
                text=doc.page_content,
                score=score,
                metadata=doc.metadata,
                doc_id=doc.metadata.get("id"),
            )
            for doc, score in results
        ]

        search_results.sort(key=lambda r: r.metadata.get("step", 0))
        return search_results

    async def compute_similarity(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """Compute cosine similarity between two texts.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score (0-1 for normalized embeddings).

        """
        embeddings = await self._embed_texts([text_a, text_b])
        # Cosine similarity (embeddings are already normalized)
        return float(np.dot(embeddings[0], embeddings[1]))

    async def similarity_matrix(
        self,
        texts_a: list[str],
        texts_b: list[str] | None = None,
    ) -> np.ndarray:
        """Compute pairwise similarity matrix between text lists.

        Args:
            texts_a: First list of texts.
            texts_b: Second list (if None, computes self-similarity).

        Returns:
            Similarity matrix of shape (len(texts_a), len(texts_b)).

        """
        emb_a = await self._embed_texts(texts_a)
        emb_b = emb_a if texts_b is None else await self._embed_texts(texts_b)

        # Cosine similarity (embeddings are already normalized)
        return np.dot(emb_a, emb_b.T)

    async def delete_session(self, session_id: str) -> int:
        """Delete all documents from a session.

        Args:
            session_id: Session to delete.

        Returns:
            Number of documents deleted.

        """
        await self._ensure_initialized()
        assert self._collection is not None

        def _delete() -> int:
            result = self._collection.remove_texts(filter={"session_id": session_id})  # type: ignore
            return int(result) if result else 0

        deleted: int = await self._run_sync(_delete)
        return deleted

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension (may trigger model load)."""
        if self._embedding_dim is None:
            model = _get_embedding_model(self.config.embedding_model)
            dim = model.get_sentence_embedding_dimension()
            self._embedding_dim = int(dim)
        return self._embedding_dim

    @property
    def cache_stats(self) -> dict[str, Any]:
        """Get embedding cache statistics."""
        return self._cache.stats

    def clear_cache(self) -> None:
        """Clear the embedding cache."""
        self._cache.clear()

    async def close(self) -> None:
        """Close database connection and shutdown executor."""
        if self._closed:
            return

        self._closed = True

        if self._db is not None:
            self._db.close()
            self._db = None
            self._collection = None

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None

    async def __aenter__(self) -> AsyncVectorStore:
        """Async context manager entry."""
        await self._ensure_initialized()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()


# Module-level singleton for shared access
_global_store: AsyncVectorStore | None = None
_global_store_lock = asyncio.Lock()


async def get_vector_store(config: VectorStoreConfig | None = None) -> AsyncVectorStore:
    """Get or create the global async vector store instance.

    Args:
        config: Configuration (only used on first call).

    Returns:
        Shared AsyncVectorStore instance.

    """
    global _global_store

    async with _global_store_lock:
        if _global_store is None:
            _global_store = AsyncVectorStore(config)
            await _global_store._ensure_initialized()
        return _global_store


async def reset_vector_store() -> None:
    """Reset the global vector store (for testing)."""
    global _global_store

    async with _global_store_lock:
        if _global_store is not None:
            await _global_store.close()
            _global_store = None
