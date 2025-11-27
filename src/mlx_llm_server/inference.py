"""MLX-based inference engine for LLM."""

from typing import Iterator, Optional
import mlx.core as mx
from mlx_lm import load, generate
from mlx_lm.utils import generate_step

from mlx_llm_server.model_loader import setup_custom_models


class MLXInferenceEngine:
    """MLX-based inference engine for language models."""

    def __init__(self, model_name: str):
        """Initialize the inference engine with a model.

        Args:
            model_name: HuggingFace model name or path
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer."""
        # Register custom models before loading
        setup_custom_models()

        print(f"Loading model: {self.model_name}")
        self.model, self.tokenizer = load(self.model_name)
        print(f"Model loaded successfully")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Generate text from a prompt.

        Args:
            prompt: Input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            stream: Whether to stream the response

        Returns:
            Generated text or iterator of text chunks
        """
        if stream:
            return self._generate_stream(prompt, max_tokens, temperature, top_p)
        else:
            return self._generate_complete(prompt, max_tokens, temperature, top_p)

    def _generate_complete(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> str:
        """Generate complete response at once."""
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            verbose=False,
        )
        return response

    def _generate_stream(
        self, prompt: str, max_tokens: int, temperature: float, top_p: float
    ) -> Iterator[str]:
        """Generate response as a stream of tokens.

        Uses token accumulation and careful UTF-8 handling to properly stream
        multi-byte characters (Korean, Chinese, Japanese) without corruption.
        This approach matches vLLM's streaming behavior.
        """
        prompt_tokens = mx.array(self.tokenizer.encode(prompt))

        token_buffer = []
        prev_text = ""

        for token, _ in zip(
            generate_step(
                prompt_tokens,
                self.model,
                temp=temperature,
                top_p=top_p,
            ),
            range(max_tokens),
        ):
            if token == self.tokenizer.eos_token_id:
                break

            # Extract token_id from tuple if needed
            token_id = token[0] if isinstance(token, tuple) else token
            token_buffer.append(token_id.item() if hasattr(token_id, 'item') else token_id)

            # Decode accumulated tokens
            # Use errors='replace' to handle incomplete UTF-8 gracefully
            try:
                current_text = self.tokenizer.decode(
                    token_buffer,
                    skip_special_tokens=True,
                    errors='replace'  # Replace incomplete sequences with �
                )
            except Exception:
                # If decode fails completely, skip this iteration
                continue

            # Calculate delta (new text only)
            new_text = current_text[len(prev_text):]

            # CRITICAL: Only yield if new_text doesn't contain replacement character
            # This prevents streaming incomplete UTF-8 sequences
            if new_text and '\ufffd' not in new_text:
                yield new_text
                prev_text = current_text
            # If replacement char is present, wait for next token to complete the character

    def apply_chat_template(self, messages: list[dict]) -> str:
        """Apply chat template to messages.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Formatted prompt string
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            # Fallback for tokenizers without chat template
            formatted = ""
            for msg in messages:
                role = msg["role"]
                content = msg["content"]
                if role == "system":
                    formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
                elif role == "user":
                    formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
                elif role == "assistant":
                    formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            formatted += "<|im_start|>assistant\n"
            return formatted
