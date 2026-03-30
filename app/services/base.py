from abc import ABC, abstractmethod
from typing import AsyncGenerator


class LLMService(ABC):
    @abstractmethod
    async def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        """Generate a text response from the LLM.

        Args:
            model: The model identifier to use for generation.
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.

        Returns:
            The generated text response.
        """

    @abstractmethod
    async def stream_generate(
        self, model: str, prompt: str, max_tokens: int = 512
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from the LLM.

        Args:
            model: The model identifier to use for generation.
            prompt: The input prompt text.
            max_tokens: Maximum number of tokens to generate.

        Yields:
            Individual text tokens as they are generated.
        """
        # Required to make this a valid async generator signature for subclasses.
        # Subclasses must use `yield` in their implementations.
        raise NotImplementedError
        yield  # noqa: unreachable — makes static analysers see this as AsyncGenerator
