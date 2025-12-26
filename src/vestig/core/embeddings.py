"""Embedding generation using sentence-transformers"""

from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """Wrapper for embedding generation with dimension validation"""

    def __init__(self, model_name: str, expected_dimension: int, normalize: bool = True):
        """
        Initialize embedding engine.

        Args:
            model_name: Name of the sentence-transformers model
            expected_dimension: Expected embedding dimension from config
            normalize: Whether to normalize embeddings for cosine similarity
        """
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.normalize = normalize
        self.model = SentenceTransformer(model_name)

        # Validate dimension on initialization
        test_embedding = self.model.encode("test", normalize_embeddings=self.normalize)
        actual_dimension = len(test_embedding)
        if actual_dimension != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, "
                f"got {actual_dimension}"
            )

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector as list of floats
        """
        embedding = self.model.encode(text, normalize_embeddings=self.normalize)
        return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of normalized embedding vectors
        """
        embeddings = self.model.encode(texts, normalize_embeddings=self.normalize)
        return embeddings.tolist()
