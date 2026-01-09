"""Embedding generation using llm or sentence-transformers"""

import json
import os
import subprocess

# Suppress HuggingFace progress bars
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_llm_model_provider(model_name: str, model_type: str = "chat") -> dict[str, str]:
    """
    Query llm CLI to determine provider information for a model.

    Args:
        model_name: Model name to look up
        model_type: "chat" or "embedding"

    Returns:
        dict with keys: provider_name, location (local/remote)
    """
    try:
        if model_type == "embedding":
            cmd = ["llm", "embed-models", "list"]
        else:
            cmd = ["llm", "models", "list"]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)

        # Parse output: "ProviderClass: model_name (aliases: ...)"
        for line in result.stdout.splitlines():
            if model_name in line:
                provider_prefix = line.split(":")[0].strip()
                if "Ollama" in provider_prefix:
                    return {"provider_name": "Ollama", "location": "local"}
                elif "OpenAI" in provider_prefix:
                    return {"provider_name": "OpenAI", "location": "remote"}
                elif "Anthropic" in provider_prefix or "Claude" in provider_prefix:
                    return {"provider_name": "Anthropic", "location": "remote"}
                else:
                    return {"provider_name": provider_prefix, "location": "unknown"}
    except Exception:
        pass

    # Fallback if detection fails
    return {"provider_name": "Unknown", "location": "unknown"}


class EmbeddingEngine:
    """Wrapper for embedding generation with dimension validation

    Supports two providers:
    - llm: Uses llm CLI (fast, 762ms with ollama)
    - sentence-transformers: Direct model loading (slow, 19s load time)
    """

    def __init__(
        self,
        model_name: str,
        expected_dimension: int,
        normalize: bool = True,
        provider: str = "llm",
        max_length: int | None = None,
        timeout: int = 60,
    ):
        """
        Initialize embedding engine.

        Args:
            model_name: Model name (format depends on provider)
                - llm: "ollama/nomic-embed-text", "ada-002", etc.
                - sentence-transformers: "BAAI/bge-m3", etc.
            expected_dimension: Expected embedding dimension from config
            normalize: Whether to normalize embeddings for cosine similarity
            provider: "llm" or "sentence-transformers"
            max_length: Maximum character length before truncation (None = no limit)
            timeout: Timeout in seconds for embedding API calls (default: 60)
        """
        self.model_name = model_name
        self.expected_dimension = expected_dimension
        self.normalize = normalize
        self.provider = provider
        self.max_length = max_length
        self.timeout = timeout

        if provider == "llm":
            # No model loading - llm CLI handles it
            # Validate that llm is available
            try:
                subprocess.run(["llm", "--version"], capture_output=True, check=True, timeout=5)
            except (subprocess.CalledProcessError, FileNotFoundError):
                raise RuntimeError(
                    "llm CLI not found. Install with: pip install llm\n"
                    "See: https://llm.datasette.io/"
                )

            # Validate dimension with a test embedding
            test_embedding = self._embed_with_llm("test")
            if len(test_embedding) != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dimension}, "
                    f"got {len(test_embedding)} for model {model_name}"
                )

        elif provider == "sentence-transformers":
            # Legacy: Load model directly (slow)
            from sentence_transformers import SentenceTransformer

            # Some models (e.g., BAAI/bge-m3) do not ship safetensors
            self.model = SentenceTransformer(model_name, model_kwargs={"use_safetensors": False})

            # Validate dimension on initialization
            test_embedding = self.model.encode("test", normalize_embeddings=self.normalize)
            actual_dimension = len(test_embedding)
            if actual_dimension != expected_dimension:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {expected_dimension}, "
                    f"got {actual_dimension}"
                )
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'llm' or 'sentence-transformers'")

    def get_provider_info(self) -> dict[str, str]:
        """
        Get provider information for the current model.

        Returns:
            dict with keys: provider_name, location (local/remote)
        """
        if self.provider == "sentence-transformers":
            return {"provider_name": "SentenceTransformers", "location": "local"}

        return get_llm_model_provider(self.model_name, model_type="embedding")

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_length if configured"""
        if self.max_length and len(text) > self.max_length:
            return text[: self.max_length]
        return text

    def _embed_with_llm(self, text: str) -> list[float]:
        """
        Generate embedding using llm CLI.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        text = self._truncate_text(text)
        try:
            result = subprocess.run(
                ["llm", "embed", "-c", text, "-m", self.model_name, "-f", "json"],
                capture_output=True,
                text=True,
                check=True,
                timeout=self.timeout,
            )
            embedding = json.loads(result.stdout)
            return embedding
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"llm embed failed: {e.stderr}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse llm output: {e}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Input text to embed

        Returns:
            Normalized embedding vector as list of floats
        """
        if self.provider == "llm":
            return self._embed_with_llm(text)
        else:  # sentence-transformers
            text = self._truncate_text(text)
            embedding = self.model.encode(
                text, normalize_embeddings=self.normalize, show_progress_bar=False
            )
            return embedding.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts to embed

        Returns:
            List of normalized embedding vectors
        """
        if self.provider == "llm":
            # llm doesn't have native batch support, so process one by one
            # Could optimize with parallel processing if needed
            return [self._embed_with_llm(text) for text in texts]
        else:  # sentence-transformers
            embeddings = self.model.encode(
                texts, normalize_embeddings=self.normalize, show_progress_bar=False
            )
            return embeddings.tolist()
