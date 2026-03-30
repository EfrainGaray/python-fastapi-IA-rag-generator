class LLMProviderError(Exception):
    """Raised when an LLM provider fails to generate a response."""


class VectorStoreError(Exception):
    """Raised when vector store operations fail."""


class DocumentNotFoundError(Exception):
    """Raised when no relevant documents are found for a query."""
