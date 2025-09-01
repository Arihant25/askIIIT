"""
Queue Manager for handling multiple model instances with user-specific chat history
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
from ollama_client import OllamaClient

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message in conversation history"""
    type: str  # 'user' or 'bot'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class UserConversation:
    """Represents a user's conversation history for a specific queue"""
    user_id: str
    queue_id: int
    messages: List[ChatMessage] = field(default_factory=list)
    last_activity: datetime = field(default_factory=datetime.now)

    def add_message(self, message: ChatMessage):
        """Add a message to the conversation"""
        self.messages.append(message)
        self.last_activity = datetime.now()

    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        """Get recent messages from the conversation"""
        return self.messages[-limit:] if len(self.messages) > limit else self.messages

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "user_id": self.user_id,
            "queue_id": self.queue_id,
            "messages": [
                {
                    "type": msg.type,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat()
                }
                for msg in self.messages
            ],
            "last_activity": self.last_activity.isoformat()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserConversation':
        """Create from dictionary"""
        messages = []
        for msg_data in data.get("messages", []):
            msg = ChatMessage(
                type=msg_data["type"],
                content=msg_data["content"],
                timestamp=datetime.fromisoformat(msg_data["timestamp"])
            )
            messages.append(msg)

        return cls(
            user_id=data["user_id"],
            queue_id=data["queue_id"],
            messages=messages,
            last_activity=datetime.fromisoformat(data["last_activity"])
        )


class QueueHandler:
    """Handles a single queue with its own model instance and user conversations"""

    def __init__(self, queue_id: int):
        self.queue_id = queue_id
        self.ollama_client = OllamaClient()
        self.user_conversations: Dict[str, UserConversation] = {}
        self.queue = asyncio.Queue()
        self.is_processing = False
        self.conversation_file = f"conversations_queue_{queue_id}.json"

        # Load existing conversations
        self._load_conversations()

    def _get_user_id_hash(self, user_identifier: str) -> str:
        """Generate a consistent hash for user identification"""
        return hashlib.md5(user_identifier.encode()).hexdigest()[:16]

    def _load_conversations(self):
        """Load conversation history from file"""
        try:
            if os.path.exists(self.conversation_file):
                with open(self.conversation_file, 'r') as f:
                    data = json.load(f)
                    for user_data in data.get("conversations", []):
                        conversation = UserConversation.from_dict(user_data)
                        self.user_conversations[conversation.user_id] = conversation
                logger.info(f"Loaded {len(self.user_conversations)} conversations for queue {self.queue_id}")
        except Exception as e:
            logger.error(f"Error loading conversations for queue {self.queue_id}: {e}")

    def _save_conversations(self):
        """Save conversation history to file"""
        try:
            data = {
                "queue_id": self.queue_id,
                "conversations": [conv.to_dict() for conv in self.user_conversations.values()],
                "last_updated": datetime.now().isoformat()
            }
            with open(self.conversation_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversations for queue {self.queue_id}: {e}")

    def get_or_create_conversation(self, user_identifier: str) -> UserConversation:
        """Get or create a conversation for a user"""
        user_id = self._get_user_id_hash(user_identifier)

        if user_id not in self.user_conversations:
            self.user_conversations[user_id] = UserConversation(
                user_id=user_id,
                queue_id=self.queue_id
            )
            logger.info(f"Created new conversation for user {user_id} in queue {self.queue_id}")

        return self.user_conversations[user_id]

    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat request"""
        try:
            user_identifier = request_data.get("user_identifier", "anonymous")
            message = request_data["message"]
            categories = request_data.get("categories", [])
            conversation_history = request_data.get("conversation_history", [])
            system_prompt = request_data.get("system_prompt", "")
            context_chunks = request_data.get("context_chunks", [])

            # Get or create user conversation
            conversation = self.get_or_create_conversation(user_identifier)

            # Add user message to conversation
            user_message = ChatMessage(type="user", content=message)
            conversation.add_message(user_message)

            # Build conversation context from stored history
            stored_messages = conversation.get_recent_messages(10)
            conversation_context = "\n".join([
                f"{'Human' if msg.type == 'user' else 'Assistant'}: {msg.content}"
                for msg in stored_messages
            ])

            # Build full system prompt with conversation history
            full_system_prompt = system_prompt
            if conversation_context:
                full_system_prompt += f"\n\nPrevious conversation:\n{conversation_context}\nPlease maintain context from this conversation."

            # Generate response
            response = await self.ollama_client.generate_response(
                prompt=message,
                context=context_chunks,
                system_prompt=full_system_prompt
            )

            # Add bot response to conversation
            bot_message = ChatMessage(type="bot", content=response)
            conversation.add_message(bot_message)

            # Save conversations periodically
            self._save_conversations()

            return {
                "response": response,
                "conversation_id": f"{conversation.user_id}_{self.queue_id}",
                "queue_id": self.queue_id,
                "model_used": self.ollama_client.chat_model
            }

        except Exception as e:
            logger.error(f"Error processing request in queue {self.queue_id}: {e}")
            return {
                "error": str(e),
                "queue_id": self.queue_id
            }

    async def process_stream_request(self, request_data: Dict[str, Any]):
        """Process a streaming chat request"""
        try:
            user_identifier = request_data.get("user_identifier", "anonymous")
            message = request_data["message"]
            categories = request_data.get("categories", [])
            conversation_history = request_data.get("conversation_history", [])
            system_prompt = request_data.get("system_prompt", "")
            context_chunks = request_data.get("context_chunks", [])

            # Get or create user conversation
            conversation = self.get_or_create_conversation(user_identifier)

            # Add user message to conversation
            user_message = ChatMessage(type="user", content=message)
            conversation.add_message(user_message)

            # Build conversation context from stored history
            stored_messages = conversation.get_recent_messages(10)
            conversation_context = "\n".join([
                f"{'Human' if msg.type == 'user' else 'Assistant'}: {msg.content}"
                for msg in stored_messages
            ])

            # Build full system prompt with conversation history
            full_system_prompt = system_prompt
            if conversation_context:
                full_system_prompt += f"\n\nPrevious conversation:\n{conversation_context}\nPlease maintain context from this conversation."

            # Generate streaming response
            response_text = ""
            async for chunk in self.ollama_client.generate_response_stream(
                prompt=message,
                context=context_chunks,
                system_prompt=full_system_prompt
            ):
                if chunk:
                    response_text += chunk
                    yield chunk

            # Add complete bot response to conversation
            if response_text:
                bot_message = ChatMessage(type="bot", content=response_text)
                conversation.add_message(bot_message)

            # Save conversations
            self._save_conversations()

        except Exception as e:
            logger.error(f"Error in streaming request for queue {self.queue_id}: {e}")
            yield f"I encountered an error: {str(e)}"

    async def start_processing(self):
        """Start processing requests from the queue"""
        self.is_processing = True
        logger.info(f"Started processing queue {self.queue_id}")

        while self.is_processing:
            try:
                # Wait for a request
                request_data = await self.queue.get()

                # Process the request
                result = await self.process_request(request_data)

                # The result will be handled by the caller
                # For now, we'll just log it
                if "error" in result:
                    logger.error(f"Request failed in queue {self.queue_id}: {result['error']}")
                else:
                    logger.info(f"Request completed in queue {self.queue_id}")

                self.queue.task_done()

            except Exception as e:
                logger.error(f"Error in queue processing {self.queue_id}: {e}")

    def stop_processing(self):
        """Stop processing requests"""
        self.is_processing = False
        logger.info(f"Stopped processing queue {self.queue_id}")


class QueueManager:
    """Manages multiple queues with model instances"""

    def __init__(self):
        self.queues_count = int(os.getenv("MODEL_QUEUES_COUNT", "3"))
        self.queue_handlers: List[QueueHandler] = []
        self.processing_tasks: List[asyncio.Task] = []
        self.is_running = False

        # Initialize queue handlers
        for i in range(self.queues_count):
            handler = QueueHandler(queue_id=i)
            self.queue_handlers.append(handler)

        logger.info(f"Initialized QueueManager with {self.queues_count} queues")

    def get_queue_for_user(self, user_identifier: str) -> QueueHandler:
        """Get the queue handler for a user (using consistent hashing)"""
        user_hash = hashlib.md5(user_identifier.encode()).hexdigest()
        queue_index = int(user_hash, 16) % self.queues_count
        return self.queue_handlers[queue_index]

    async def start_queues(self):
        """Start all queue processing tasks"""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Starting all queue processing tasks")

        for handler in self.queue_handlers:
            task = asyncio.create_task(handler.start_processing())
            self.processing_tasks.append(task)

    def stop_queues(self):
        """Stop all queue processing"""
        if not self.is_running:
            return

        self.is_running = False
        logger.info("Stopping all queue processing")

        for handler in self.queue_handlers:
            handler.stop_processing()

        # Cancel all processing tasks
        for task in self.processing_tasks:
            if not task.done():
                task.cancel()

    async def process_chat_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chat request using the appropriate queue"""
        user_identifier = request_data.get("user_identifier", "anonymous")
        queue_handler = self.get_queue_for_user(user_identifier)

        logger.info(f"Routing request for user {user_identifier} to queue {queue_handler.queue_id}")

        # Process directly (not using the queue for now, but could be changed to async queue)
        return await queue_handler.process_request(request_data)

    async def process_stream_request(self, request_data: Dict[str, Any]):
        """Process a streaming chat request using the appropriate queue"""
        user_identifier = request_data.get("user_identifier", "anonymous")
        queue_handler = self.get_queue_for_user(user_identifier)

        logger.info(f"Routing streaming request for user {user_identifier} to queue {queue_handler.queue_id}")

        # Process streaming request
        async for chunk in queue_handler.process_stream_request(request_data):
            yield chunk

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about all queues"""
        stats = {
            "total_queues": self.queues_count,
            "is_running": self.is_running,
            "queues": []
        }

        for handler in self.queue_handlers:
            queue_stat = {
                "queue_id": handler.queue_id,
                "active_conversations": len(handler.user_conversations),
                "is_processing": handler.is_processing,
                "queue_size": handler.queue.qsize()
            }
            stats["queues"].append(queue_stat)

        return stats


# Global queue manager instance
queue_manager = QueueManager()
