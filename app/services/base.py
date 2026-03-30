from abc import ABC, abstractmethod


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
