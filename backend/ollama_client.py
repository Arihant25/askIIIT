"""
HF Text Generation Inference client for Qwen models
Compatible with OpenAI-style /v1/chat/completions API
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
            "OLLAMA_BASE_URL", "http://localhost:8000"
        )
        self.chat_model = os.getenv("OLLAMA_CHAT_MODEL", "Qwen/Qwen3-1.7B")
        self.timeout = httpx.Timeout(300.0)  # 5 minutes for long operations

    async def check_connection(self) -> bool:
        """Check if HF TGI server is running"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"HF TGI connection failed: {e}")
            return False

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models in HF TGI"""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
                response = await client.get(f"{self.base_url}/v1/models")
                response.raise_for_status()
                data = response.json()
                return data.get("data", [])
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []

    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure model is available (HF TGI doesn't support pulling, assume pre-loaded)"""
        try:
            # For HF TGI, the model is assumed to be already running
            # Just verify the service is available
            is_available = await self.check_connection()
            if is_available:
                # logger.info(f"HF TGI service is available with model ready")
                return True
            else:
                logger.error(f"HF TGI service not available")
                return False
        except Exception as e:
            logger.error(f"Failed to verify model availability: {e}")
            return False

    async def generate_response(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        context_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """Generate response using Qwen model via HF TGI /v1/chat/completions API"""
        try:
            # Ensure chat model is available
            if not await self.ensure_model_available(self.chat_model):
                raise Exception(f"Chat model {self.chat_model} not available")

            # Build messages in OpenAI format
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            if context and context_metadata:
                context_text = "Context from the following documents:\n"
                # Limit to 5 context chunks and track their filenames
                for i, (ctx, metadata) in enumerate(zip(context[:5], context_metadata[:5])):
                    filename = metadata.get('filename', 'Unknown')
                    context_text += f"{i+1}. [From: {filename}] {ctx}\n\n"
                messages.append({"role": "system", "content": context_text})
            elif context:
                context_text = "Context:\n"
                # Limit to 5 context chunks
                for i, ctx in enumerate(context[:5]):
                    context_text += f"{i+1}. {ctx}\n"
                messages.append({"role": "system", "content": context_text})
            
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.chat_model,
                        "messages": messages,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "stream": False,
                    },
                )
                response.raise_for_status()
                data = response.json()

                if "choices" in data and len(data["choices"]) > 0:
                    return data["choices"][0]["message"]["content"].strip()
                else:
                    logger.error("No response in HF TGI API response")
                    return (
                        "I apologize, but I couldn't generate a response at this time."
                    )

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I encountered an error while processing your request: {str(e)}"

    async def generate_response_stream(
        self,
        prompt: str,
        context: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        context_metadata: Optional[List[Dict[str, Any]]] = None,
    ):
        """Generate streaming response using Qwen model via HF TGI /v1/chat/completions API"""
        try:
            # Ensure chat model is available
            if not await self.ensure_model_available(self.chat_model):
                raise Exception(f"Chat model {self.chat_model} not available")

            # Build messages in OpenAI format
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            if context and context_metadata:
                context_text = "Context from the following documents:\n"
                # Limit to 5 context chunks and track their filenames
                for i, (ctx, metadata) in enumerate(zip(context[:5], context_metadata[:5])):
                    filename = metadata.get('filename', 'Unknown')
                    context_text += f"{i+1}. [From: {filename}] {ctx}\n\n"
                messages.append({"role": "system", "content": context_text})
            elif context:
                context_text = "Context:\n"
                # Limit to 5 context chunks
                for i, ctx in enumerate(context[:5]):
                    context_text += f"{i+1}. {ctx}\n"
                messages.append({"role": "system", "content": context_text})
            
            messages.append({"role": "user", "content": prompt})

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.chat_model,
                        "messages": messages,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 4096,
                        "stream": True,
                    },
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            # Remove 'data: ' prefix if present
                            if line.startswith("data: "):
                                line = line[6:]
                            
                            # Skip [DONE] messages
                            if line == "[DONE]":
                                break
                            
                            try:
                                data = json.loads(line)
                                if "choices" in data and len(data["choices"]) > 0:
                                    choice = data["choices"][0]
                                    if "delta" in choice and "content" in choice["delta"]:
                                        content = choice["delta"]["content"]
                                        if content:
                                            # Yield each chunk immediately
                                            for char in content:
                                                yield char
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            yield f"I encountered an error while processing your request: {str(e)}"

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
            system_prompt = (
                "You are a document categorization assistant. "
                "Categorize the following document into exactly one of these categories: "
                "faculty, student, hostel, academics, mess. "
                "Respond with only the category name, nothing else."
            )

            prompt = (
                f"Filename: {filename}\n\n"
                f"Content: {text}\n\n"
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

    def extract_referenced_files(self, response: str, available_files: List[str]) -> List[str]:
        """Extract which files were actually referenced from the LLM response"""
        try:
            # Look for files mentioned in the response
            referenced_files = []
            response_lower = response.lower()
            
            for filename in available_files:
                # Check if filename (case-insensitive) appears in the response
                if filename.lower() in response_lower:
                    referenced_files.append(filename)
            
            logger.info(f"Extracted {len(referenced_files)} referenced files from response")
            return referenced_files
        except Exception as e:
            logger.error(f"Error extracting referenced files: {e}")
            # Return all files if extraction fails
            return available_files


# Global HF TGI client instance
ollama_client = OllamaClient()
