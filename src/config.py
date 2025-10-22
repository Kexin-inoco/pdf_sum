"""
Configuration management module.
Simple environment variable-based configuration.
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """
    Simple configuration from environment variables.
    All settings have sensible defaults.
    """
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        pass
    
    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OPENAI_API_KEY not found. "
                "Please set it in .env file or environment variables."
            )
        return key
    
    @openai_api_key.setter
    def openai_api_key(self, value: str):
        """Set OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = value
    
    @property
    def summarization_strategy(self) -> str:
        """Get summarization strategy (stuff, map_reduce, refine)."""
        return os.getenv("SUMMARIZATION_STRATEGY", "map_reduce")
    
    @property
    def llm_model(self) -> str:
        """Get LLM model name."""
        return os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    
    @property
    def temperature(self) -> float:
        """Get LLM temperature."""
        return float(os.getenv("TEMPERATURE", "0.3"))
    
    @property
    def max_tokens(self) -> int:
        """Get max tokens for LLM response."""
        return int(os.getenv("MAX_TOKENS", "500"))
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(model={self.llm_model}, strategy={self.summarization_strategy})"


# Global configuration instance
config = Config()

