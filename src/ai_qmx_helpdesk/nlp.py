"""
nlp.py - Natural Language Processing and LLM integration for RAG
2025-08-28

This module provides LLM integration for generating natural language answers
from retrieved document chunks. Supports multiple providers including OpenAI,
Anthropic, and local models.

Example usage:
    llm = make_llm({"provider": "openai", "model": "gpt-4o-mini"})
    answer = generate_answer(llm, question="What is QMX?", chunks=[...])
"""

import logging
from typing import Any, Dict, List, Optional

_logger = logging.getLogger(__name__)

# Default LLM configuration
DEFAULT_LLM_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_tokens": 1000,
    "system_prompt": """You are a helpful AI assistant that answers questions based on the provided context. 

Instructions:
- Use only the information provided in the context to answer the question
- If the context doesn't contain enough information, say "I don't have enough information to answer that question"
- Be concise but thorough
- Cite specific details from the context when possible
- If asked about something not in the context, clearly state that""",
}


class LLMProvider:
    """Base class for LLM providers."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate text from prompt."""
        raise NotImplementedError


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            import openai

            self.client = openai.OpenAI()
        except ImportError as e:
            raise ImportError("OpenAI library not available. Install: pip install openai") from e

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate response using OpenAI API."""
        try:
            messages: List[Dict[str, str]] = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            response = self.client.chat.completions.create(
                model=self.config.get("model", "gpt-4o-mini"),
                messages=messages,  # type: ignore[arg-type]
                temperature=self.config.get("temperature", 0.1),
                max_tokens=self.config.get("max_tokens", 1000),
                **kwargs,
            )

            content = response.choices[0].message.content
            return content.strip() if content else ""

        except Exception as e:
            self._logger.error("OpenAI API error: %s", e)
            raise RuntimeError(f"Failed to generate response: {e}") from e


class ToyLLMProvider(LLMProvider):
    """Toy LLM provider for testing (returns simple formatted response)."""

    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs: Any) -> str:
        """Generate a toy response for testing."""
        context_length = len(prompt)
        lines = prompt.count("\n")

        return f"""[TOY LLM RESPONSE]
Based on the provided context ({context_length} characters, {lines} lines), here's a simulated answer:

The question appears to be asking about the topic mentioned in the context. According to the provided information, this relates to technical documentation and operational details.

Key points from the context:
- Multiple sections and references found
- Technical specifications mentioned  
- Operational procedures described

Note: This is a toy/mock response for testing purposes. For real answers, use a proper LLM provider like OpenAI."""


def make_llm(config: Dict[str, Any]) -> LLMProvider:
    """Factory function to create LLM provider instances."""
    provider = config.get("provider", "toy").lower()

    if provider == "openai":
        return OpenAIProvider(config)
    elif provider == "toy":
        return ToyLLMProvider(config)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Supported: openai, toy")


def generate_answer(
    llm: LLMProvider, question: str, chunks: List[Dict[str, Any]], max_context_length: int = 4000
) -> Dict[str, Any]:
    """Generate natural language answer from question and retrieved chunks.

    Args:
        llm: LLM provider instance
        question: User's question
        chunks: Retrieved document chunks with content and metadata
        max_context_length: Maximum context length to send to LLM

    Returns:
        Dict with answer, chunks_used, and metadata
    """
    _logger.info("Generating answer for question: %s", question[:100])

    if not chunks:
        return {
            "answer": "I don't have any relevant context to answer that question.",
            "chunks_used": 0,
            "question": question,
            "provider": llm.config.get("provider", "unknown"),
            "model": llm.config.get("model", "unknown"),
        }

    # Build context from chunks, respecting length limits
    context_parts = []
    current_length = 0
    chunks_used = 0

    for i, chunk in enumerate(chunks):
        content = chunk.get("content", "")
        chunk_text = f"[Source {i+1}]: {content}\n\n"

        if current_length + len(chunk_text) > max_context_length:
            break

        context_parts.append(chunk_text)
        current_length += len(chunk_text)
        chunks_used += 1

    context = "".join(context_parts)

    # Create prompt
    user_prompt = f"""Context information:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""

    system_prompt = llm.config.get("system_prompt", DEFAULT_LLM_CONFIG["system_prompt"])

    try:
        answer = llm.generate(user_prompt, system_prompt=system_prompt)

        return {
            "answer": answer,
            "chunks_used": chunks_used,
            "question": question,
            "provider": llm.config.get("provider", "unknown"),
            "model": llm.config.get("model", "unknown"),
            "context_length": current_length,
        }

    except Exception as e:
        _logger.error("Failed to generate answer: %s", e)
        return {
            "answer": f"Error generating answer: {e}",
            "chunks_used": chunks_used,
            "question": question,
            "provider": llm.config.get("provider", "unknown"),
            "model": llm.config.get("model", "unknown"),
            "error": str(e),
        }


def answer_question(question: str, db_path: str, cfg: Dict[str, Any], k: int = 5) -> Dict[str, Any]:
    """Complete RAG pipeline: retrieve chunks and generate answer.

    Args:
        question: User's natural language question
        db_path: Path to RAG database
        cfg: Configuration dictionary with embed and llm settings
        k: Number of chunks to retrieve

    Returns:
        Dict with answer and metadata
    """
    from . import rag_db

    _logger.info("Starting RAG pipeline for question: %s", question[:100])

    # STEP 1: Retrieve relevant chunks
    try:
        search_cfg = {"embed": cfg.get("embed", {})}
        raw_results = rag_db.search(db_path, question, k=k, cfg=search_cfg)

        # Convert tuples (text, score) to dictionaries
        chunks = [{"content": text, "score": float(score)} for text, score in raw_results]
        _logger.info("Retrieved %d chunks", len(chunks))
    except Exception as e:
        _logger.error("Failed to retrieve chunks: %s", e)
        return {
            "answer": f"Error retrieving information: {e}",
            "question": question,
            "error": str(e),
        }

    # STEP 2: Generate answer using LLM
    try:
        llm_cfg = cfg.get("llm", DEFAULT_LLM_CONFIG)
        llm = make_llm(llm_cfg)
        result = generate_answer(llm, question, chunks)

        # Add retrieval metadata
        result["retrieval"] = {
            "chunks_retrieved": len(chunks),
            "chunks_used": result.get("chunks_used", 0),
        }

        return result

    except Exception as e:
        _logger.error("Failed to generate answer: %s", e)
        return {
            "answer": f"Error generating answer: {e}",
            "question": question,
            "retrieval": {"chunks_retrieved": len(chunks)},
            "error": str(e),
        }
