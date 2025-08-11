"""
Ollama client capabilities for Qwen models
"""

import os
import httpx
import logging
from typing import List, Dict, Any, Optional
import json
import asyncio

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "qwen3:0.6B")
        self.timeout = httpx.Timeout(300.0)  # 5 minutes for long operations

    async def check_connection(self) -> bool:
        """Check if Ollama server is running"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()
                data = response.json()
                return data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pull if necessary"""
        try:
            models = await self.list_models()
            available_models = [model["name"] for model in models]

            if model_name in available_models:
                logger.info(f"Model {model_name} is already available")
                return True

            logger.info(f"Pulling model {model_name}...")
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(600.0)
            ) as client:  # 10 minutes for model pull
                async with client.stream(
                    "POST", f"{self.base_url}/api/pull", json={"name": model_name}
                ) as response:
                    response.raise_for_status()
                    last_status = ""
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                data = json.loads(line)
                                if "status" in data:
                                    current_status = data['status']
                                    # Only log when status changes to reduce spam
                                    if current_status != last_status:
                                        logger.info(f"Pull status: {current_status}")
                                        last_status = current_status
                                    
                                    if current_status == "success":
                                        logger.info(f"Model {model_name} pulled successfully")
                                        return True
                            except json.JSONDecodeError:
                                continue

            return False
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            return False

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate response using Qwen chat model"""
        try:
            # Ensure chat model is available
            if not await self.ensure_model_available(self.chat_model):
                raise Exception(f"Chat model {self.chat_model} not available")

            # Build the full prompt
            full_prompt = ""

            if system_prompt:
                full_prompt += f"System: {system_prompt}\n\n"

            if context:
                full_prompt += "Context:\n"
                # Limit to 5 context chunks
                for i, ctx in enumerate(context[:5]):
                    full_prompt += f"{i+1}. {ctx}\n"
                full_prompt += "\n"

            full_prompt += f"User: {prompt}\n\nAssistant:"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": self.chat_model,
                        "prompt": full_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 2048,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

                if "response" in data:
                    return data["response"].strip()
                else:
                    logger.error("No response in Ollama API response")
                    return (
                        "I apologize, but I couldn't generate a response at this time."
                    )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    async def summarize_text(self, text: str, max_length: int = 200) -> str:
        """Generate summary of text using Qwen model"""
        try:
            # Truncate text if too long
            if len(text) > 4000:
                text = text[:4000] + "..."

            system_prompt = (
                "You are a helpful assistant that creates concise summaries. "
                f"Summarize the following text in no more than {max_length} characters. "
                "Focus on the key information and main purpose of the document."
            )

            prompt = f"Please summarize this text:\n\n{text}"

            summary = await self.generate_response(
                prompt=prompt, system_prompt=system_prompt
            )

            # Ensure summary is within length limit
            if len(summary) > max_length:
                summary = summary[: max_length - 3] + "..."

            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Document summary (auto-generated): {text[:100]}..."

    async def categorize_document(self, filename: str, text: str) -> str:
        """Automatically categorize document using Qwen model"""
        try:
            # Use first 1000 characters for categorization
            sample_text = text[:1000] if len(text) > 1000 else text

            system_prompt = (
                "You are a document categorization assistant. "
                "Categorize the following document into exactly one of these categories: "
                "faculty, student, hostel, academics, mess. "
                "Respond with only the category name, nothing else."
            )

            prompt = (
                f"Filename: {filename}\n\n"
                f"Content sample: {sample_text}\n\n"
                "Category:"
            )

            category = await self.generate_response(
                prompt=prompt, system_prompt=system_prompt
            )

            # Clean and validate category
            category = category.lower().strip()
            valid_categories = ["faculty", "student",
                                "hostel", "academics", "mess"]

            if category in valid_categories:
                return category
            else:
                # Fallback to keyword-based categorization
                filename_lower = filename.lower()
                text_lower = text.lower()

                if any(
                    word in filename_lower or word in text_lower
                    for word in ["faculty", "staff", "professor", "teacher"]
                ):
                    return "faculty"
                elif any(
                    word in filename_lower or word in text_lower
                    for word in ["student", "admission", "grade", "course"]
                ):
                    return "student"
                elif any(
                    word in filename_lower or word in text_lower
                    for word in ["hostel", "accommodation", "room", "residence"]
                ):
                    return "hostel"
                elif any(
                    word in filename_lower or word in text_lower
                    for word in ["mess", "food", "dining", "meal"]
                ):
                    return "mess"
                else:
                    return "academics"  # Default category

        except Exception as e:
            logger.error(f"Error categorizing document: {e}")
            return "academics"  # Default fallback


# Global Ollama client instance
ollama_client = OllamaClient()
