"""
test_nlp.py - Test cases for NLP and answer generation functionality
2025-08-28
"""

import pytest
from unittest.mock import Mock, patch
from typing import Any

from ai_qmx_helpdesk.nlp import (
    make_llm,
    generate_answer,
    answer_question,
    ToyLLMProvider,
    OpenAIProvider,
    DEFAULT_LLM_CONFIG,
)


class TestLLMProviders:
    """Test LLM provider creation and functionality."""

    def test_make_llm_toy_provider(self) -> None:
        """Test creating toy LLM provider."""
        config = {"provider": "toy"}
        llm = make_llm(config)
        assert isinstance(llm, ToyLLMProvider)

    def test_make_llm_openai_provider(self) -> None:
        """Test creating OpenAI provider (mock)."""
        with patch("openai.OpenAI"):
            config = {"provider": "openai", "model": "gpt-4"}
            llm = make_llm(config)
            assert isinstance(llm, OpenAIProvider)

    def test_make_llm_unknown_provider(self) -> None:
        """Test error for unknown provider."""
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            make_llm({"provider": "unknown"})

    def test_toy_llm_generate(self) -> None:
        """Test toy LLM response generation."""
        llm = ToyLLMProvider({"provider": "toy"})
        response = llm.generate("Test prompt with some context")

        assert "[TOY LLM RESPONSE]" in response
        assert "characters" in response
        assert "testing purposes" in response


class TestAnswerGeneration:
    """Test answer generation from chunks."""

    def test_generate_answer_no_chunks(self) -> None:
        """Test generating answer with no chunks."""
        llm = ToyLLMProvider({"provider": "toy", "model": "test"})
        result = generate_answer(llm, "What is test?", [])

        assert "don't have any relevant context" in result["answer"]
        assert result["chunks_used"] == 0
        assert result["question"] == "What is test?"
        assert result["provider"] == "toy"

    def test_generate_answer_with_chunks(self) -> None:
        """Test generating answer with chunks."""
        llm = ToyLLMProvider({"provider": "toy", "model": "test"})
        chunks = [
            {"content": "QMX is a radio transceiver"},
            {"content": "It operates on multiple bands"},
            {"content": "Built by QRP Labs"},
        ]

        result = generate_answer(llm, "What is QMX?", chunks)

        assert "[TOY LLM RESPONSE]" in result["answer"]
        assert result["chunks_used"] == 3
        assert result["question"] == "What is QMX?"
        assert result["provider"] == "toy"
        assert "context_length" in result

    def test_generate_answer_context_limit(self) -> None:
        """Test context length limiting."""
        llm = ToyLLMProvider({"provider": "toy"})

        # Create chunks that exceed limit
        large_chunk = "x" * 3000
        chunks = [{"content": large_chunk}, {"content": "This should be truncated"}]

        result = generate_answer(llm, "Test?", chunks, max_context_length=100)

        # Should only use first chunk (partially)
        assert result["chunks_used"] <= 1
        assert result["context_length"] <= 100


class TestRAGPipeline:
    """Test complete RAG pipeline integration."""

    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_answer_question_success(self, mock_search: Any) -> None:
        """Test successful answer generation pipeline."""
        # Mock retrieval results - rag_db.search returns tuples (text, score)
        mock_search.return_value = [("QMX is a transceiver", 0.9), ("It's made by QRP Labs", 0.8)]

        config = {"embed": {"provider": "toy"}, "llm": {"provider": "toy", "model": "test"}}

        result = answer_question("What is QMX?", "test.db", config, k=5)

        # Check result structure
        assert result["question"] == "What is QMX?"
        assert result["provider"] == "toy"
        assert result["model"] == "test"
        assert "answer" in result
        assert "retrieval" in result
        assert result["retrieval"]["chunks_retrieved"] == 2

        # Verify rag_db.search was called correctly
        mock_search.assert_called_once_with(
            "test.db", "What is QMX?", k=5, cfg={"embed": {"provider": "toy"}}
        )

    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_answer_question_retrieval_error(self, mock_search: Any) -> None:
        """Test handling of retrieval errors."""
        mock_search.side_effect = Exception("Database error")

        config = {"embed": {}, "llm": {"provider": "toy"}}
        result = answer_question("Test?", "test.db", config)

        assert "Error retrieving information" in result["answer"]
        assert "error" in result

    @patch("ai_qmx_helpdesk.rag_db.search")
    def test_answer_question_generation_error(self, mock_search: Any) -> None:
        """Test handling of answer generation errors."""
        mock_search.return_value = [("test", 0.9)]

        # Use invalid LLM config to trigger error
        config = {"embed": {"provider": "toy"}, "llm": {"provider": "invalid"}}

        result = answer_question("Test?", "test.db", config)

        assert "Error generating answer" in result["answer"]
        assert "error" in result


class TestOpenAIIntegration:
    """Test OpenAI provider integration (mocked)."""

    @patch("openai.OpenAI")
    def test_openai_provider_generate(self, mock_openai_class: Any) -> None:
        """Test OpenAI provider response generation."""
        # Mock OpenAI client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "This is a test response from GPT"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        # Create provider and test
        config = {"provider": "openai", "model": "gpt-4", "temperature": 0.2}
        provider = OpenAIProvider(config)

        result = provider.generate("Test prompt", system_prompt="Test system")

        assert result == "This is a test response from GPT"

        # Verify API call
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Test system"},
                {"role": "user", "content": "Test prompt"},
            ],
            temperature=0.2,
            max_tokens=1000,
        )

    @patch("openai.OpenAI")
    def test_openai_provider_api_error(self, mock_openai_class: Any) -> None:
        """Test OpenAI API error handling."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider({"provider": "openai"})

        with pytest.raises(RuntimeError, match="Failed to generate response"):
            provider.generate("Test prompt")


class TestConfiguration:
    """Test configuration handling."""

    def test_default_llm_config(self) -> None:
        """Test default LLM configuration values."""
        assert DEFAULT_LLM_CONFIG["provider"] == "openai"
        assert DEFAULT_LLM_CONFIG["model"] == "gpt-4o-mini"
        assert DEFAULT_LLM_CONFIG["temperature"] == 0.1
        assert DEFAULT_LLM_CONFIG["max_tokens"] == 1000
        assert "system_prompt" in DEFAULT_LLM_CONFIG

    def test_config_override(self) -> None:
        """Test configuration override in providers."""
        custom_config = {
            "provider": "toy",
            "temperature": 0.8,
            "max_tokens": 2000,
            "custom_field": "test",
        }

        provider = ToyLLMProvider(custom_config)
        assert provider.config["temperature"] == 0.8
        assert provider.config["max_tokens"] == 2000
        assert provider.config["custom_field"] == "test"


if __name__ == "__main__":
    pytest.main([__file__])
