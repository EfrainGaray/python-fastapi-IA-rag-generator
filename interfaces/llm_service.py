class LLMService:
    def generate(self, model: str, prompt: str, max_tokens: int = 512) -> str:
        raise NotImplementedError("This method should be overridden by subclasses")
