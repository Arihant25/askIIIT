"""
Main load testing file using Locust for askIIIT application.
Tests various endpoints with realistic user behavior patterns.
"""

from locust import HttpUser, task, between, events
import json
import random
import time
import uuid
from typing import Dict, Any, Optional, List
import logging

from config import config, test_data, MockAuthHeaders

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AskIIITUser(HttpUser):
    """
    Base user class for askIIIT application load testing.
    Simulates realistic user behavior patterns.
    """
    
    wait_time = between(config.min_wait / 1000, config.max_wait / 1000)
    host = config.base_url
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_id = None
        self.conversation_history = []
        self.is_admin = False
        self.user_info = None
        
    def on_start(self):
        """Initialize user session"""
        # Randomly assign admin status (10% chance)
        self.is_admin = random.random() < 0.1
        self.user_info = test_data.generate_user_info(self.is_admin)
        
        # Try to authenticate (health check serves as auth verification)
        self.health_check()
        
        logger.info(f"User started: {self.user_info['email']}, Admin: {self.is_admin}")
    
    def health_check(self):
        """Basic health check to verify service availability"""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(10)
    def search_documents(self):
        """Test document search functionality"""
        query_data = test_data.generate_search_query()
        
        with self.client.post(
            "/api/search",
            json=query_data,
            headers=MockAuthHeaders.get_headers(),
            catch_response=True,
            name="Document Search"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Validate response structure
                    if "documents" in result and "content_chunks" in result:
                        response.success()
                        logger.debug(f"Search successful: {len(result['documents'])} docs found")
                    else:
                        response.failure("Invalid response structure")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Search failed: {response.status_code}")

    @task(15)
    def chat_with_documents(self):
        """Test chat functionality with conversation context"""
        # Generate chat message
        chat_data = test_data.generate_chat_message(self.conversation_id)
        
        # Add conversation history if exists
        if self.conversation_history:
            chat_data["conversation_history"] = self.conversation_history[-10:]  # Last 10 messages
        
        with self.client.post(
            "/api/chat",
            json=chat_data,
            headers=MockAuthHeaders.get_headers(),
            catch_response=True,
            name="Chat"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "response" in result:
                        # Update conversation state
                        if not self.conversation_id:
                            self.conversation_id = result.get("conversation_id", str(uuid.uuid4()))
                        
                        # Add to conversation history
                        self.conversation_history.append({
                            "type": "user",
                            "content": chat_data["message"]
                        })
                        self.conversation_history.append({
                            "type": "bot", 
                            "content": result["response"]
                        })
                        
                        # Limit history size
                        if len(self.conversation_history) > 20:
                            self.conversation_history = self.conversation_history[-20:]
                        
                        response.success()
                        logger.debug(f"Chat successful: conversation_id={self.conversation_id}")
                    else:
                        response.failure("No response in chat result")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Chat failed: {response.status_code}")

    @task(8)
    def streaming_chat(self):
        """Test streaming chat functionality"""
        chat_data = test_data.generate_chat_message(self.conversation_id)
        
        if self.conversation_history:
            chat_data["conversation_history"] = self.conversation_history[-5:]  # Last 5 for streaming
        
        start_time = time.time()
        
        with self.client.post(
            "/api/chat/stream",
            json=chat_data,
            headers=MockAuthHeaders.get_headers(),
            stream=True,
            catch_response=True,
            name="Streaming Chat"
        ) as response:
            if response.status_code == 200:
                try:
                    chunks_received = 0
                    total_content = ""
                    
                    for line in response.iter_lines():
                        if line:
                            line_str = line.decode('utf-8')
                            if line_str.startswith('data: '):
                                try:
                                    data = json.loads(line_str[6:])  # Remove 'data: ' prefix
                                    chunks_received += 1
                                    
                                    if data.get("type") == "content":
                                        total_content += data.get("content", "")
                                        
                                    if data.get("is_final"):
                                        break
                                        
                                except json.JSONDecodeError:
                                    continue
                    
                    if chunks_received > 0:
                        response.success()
                        logger.debug(f"Streaming chat successful: {chunks_received} chunks")
                    else:
                        response.failure("No valid chunks received")
                        
                except Exception as e:
                    response.failure(f"Streaming error: {str(e)}")
            else:
                response.failure(f"Streaming chat failed: {response.status_code}")

    @task(5)
    def list_documents(self):
        """Test document listing"""
        params = {}
        
        # Sometimes filter by categories
        if random.random() < 0.4:
            categories = random.sample(test_data.categories, random.randint(1, 2))
            params["categories"] = ",".join(categories)
        
        # Random pagination
        params["limit"] = random.randint(10, 50)
        params["offset"] = random.randint(0, 20)
        
        with self.client.get(
            "/api/documents",
            params=params,
            headers=MockAuthHeaders.get_headers(),
            catch_response=True,
            name="List Documents"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "documents" in result:
                        response.success()
                        logger.debug(f"Document list successful: {len(result['documents'])} docs")
                    else:
                        response.failure("Invalid document list response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Document list failed: {response.status_code}")

    @task(3)
    def get_categories(self):
        """Test categories endpoint"""
        with self.client.get(
            "/api/categories",
            headers=MockAuthHeaders.get_headers(),
            catch_response=True,
            name="Get Categories"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "categories" in result:
                        response.success()
                    else:
                        response.failure("Invalid categories response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Categories failed: {response.status_code}")

    @task(2)
    def get_stats(self):
        """Test stats endpoint"""
        with self.client.get(
            "/api/stats",
            headers=MockAuthHeaders.get_headers(),
            catch_response=True,
            name="Get Stats"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "total_documents" in result:
                        response.success()
                    else:
                        response.failure("Invalid stats response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Stats failed: {response.status_code}")

    @task(4)
    def health_and_system_check(self):
        """Test health and system endpoints"""
        with self.client.get(
            "/health",
            catch_response=True,
            name="Health Check"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "status" in result:
                        response.success()
                    else:
                        response.failure("Invalid health response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            else:
                response.failure(f"Health check failed: {response.status_code}")


class AdminUser(AskIIITUser):
    """
    Admin user class with additional admin-specific tasks.
    Tests admin endpoints with appropriate privileges.
    """
    
    weight = 1  # Lower weight means fewer admin users
    
    def on_start(self):
        """Initialize admin user session"""
        self.is_admin = True
        self.user_info = test_data.generate_user_info(is_admin=True)
        self.health_check()
        logger.info(f"Admin user started: {self.user_info['email']}")

    @task(3)
    def get_system_info(self):
        """Test admin system info endpoint"""
        with self.client.get(
            "/api/admin/system-info",
            headers=MockAuthHeaders.get_admin_headers(),
            catch_response=True,
            name="Admin System Info"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "documents" in result and "backend" in result:
                        response.success()
                        logger.debug("Admin system info successful")
                    else:
                        response.failure("Invalid system info response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"System info failed: {response.status_code}")

    @task(2)
    def get_admin_logs(self):
        """Test admin logs endpoint"""
        params = {
            "lines": random.randint(50, 200),
            "level": random.choice(["INFO", "ERROR", "WARNING", None])
        }
        
        with self.client.get(
            "/api/admin/logs",
            params={k: v for k, v in params.items() if v is not None},
            headers=MockAuthHeaders.get_admin_headers(),
            catch_response=True,
            name="Admin Logs"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "logs" in result:
                        response.success()
                        logger.debug(f"Admin logs successful: {len(result['logs'])} entries")
                    else:
                        response.failure("Invalid logs response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"Admin logs failed: {response.status_code}")

    @task(1)
    def get_users(self):
        """Test admin users endpoint"""
        with self.client.get(
            "/api/admin/users",
            headers=MockAuthHeaders.get_admin_headers(),
            catch_response=True,
            name="Admin Users"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "admin_users" in result:
                        response.success()
                    else:
                        response.failure("Invalid users response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"Admin users failed: {response.status_code}")

    @task(1)
    def rag_diagnostics(self):
        """Test RAG diagnostics endpoint"""
        params = {"query": random.choice(test_data.sample_queries)}
        
        with self.client.get(
            "/api/admin/debug/rag-diagnostics",
            params=params,
            headers=MockAuthHeaders.get_admin_headers(),
            catch_response=True,
            name="RAG Diagnostics"
        ) as response:
            if response.status_code == 200:
                try:
                    result = response.json()
                    if "collections" in result and "rag_results" in result:
                        response.success()
                    else:
                        response.failure("Invalid diagnostics response")
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
            elif response.status_code == 403:
                response.failure("Admin access denied")
            else:
                response.failure(f"RAG diagnostics failed: {response.status_code}")


class HeavyUser(AskIIITUser):
    """
    Heavy user class that simulates intensive usage patterns.
    Tests system under high load conditions.
    """
    
    weight = 1  # Lower weight for heavy users
    wait_time = between(0.1, 0.5)  # Much shorter wait times
    
    def on_start(self):
        super().on_start()
        logger.info(f"Heavy user started: {self.user_info['email']}")

    @task(20)
    def rapid_search(self):
        """Perform rapid consecutive searches"""
        for _ in range(random.randint(3, 7)):
            query_data = test_data.generate_search_query()
            
            with self.client.post(
                "/api/search",
                json=query_data,
                headers=MockAuthHeaders.get_headers(),
                catch_response=True,
                name="Rapid Search"
            ) as response:
                if response.status_code != 200:
                    response.failure(f"Rapid search failed: {response.status_code}")
                    break
                else:
                    response.success()
            
            time.sleep(0.1)  # Brief pause between requests

    @task(15)
    def extended_conversation(self):
        """Simulate an extended conversation session"""
        conversation_length = random.randint(8, 15)
        
        for i in range(conversation_length):
            chat_data = test_data.generate_chat_message(self.conversation_id)
            
            if self.conversation_history:
                chat_data["conversation_history"] = self.conversation_history
            
            with self.client.post(
                "/api/chat",
                json=chat_data,
                headers=MockAuthHeaders.get_headers(),
                catch_response=True,
                name="Extended Chat"
            ) as response:
                if response.status_code == 200:
                    try:
                        result = response.json()
                        if "response" in result:
                            # Update conversation state
                            if not self.conversation_id:
                                self.conversation_id = result.get("conversation_id")
                            
                            self.conversation_history.append({
                                "type": "user",
                                "content": chat_data["message"]
                            })
                            self.conversation_history.append({
                                "type": "bot",
                                "content": result["response"]
                            })
                            
                            response.success()
                        else:
                            response.failure("No response in chat result")
                            break
                    except json.JSONDecodeError:
                        response.failure("Invalid JSON response")
                        break
                else:
                    response.failure(f"Extended chat failed: {response.status_code}")
                    break
            
            time.sleep(random.uniform(0.2, 0.8))  # Simulate thinking time


# Event handlers for custom metrics and logging
@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, context, **kwargs):
    """Log detailed request information"""
    if exception:
        logger.error(f"Request failed: {name} - {exception}")
    elif response_time > 5000:  # Log slow requests (>5 seconds)
        logger.warning(f"Slow request: {name} - {response_time}ms")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Log test start information"""
    logger.info(f"Load test started with {environment.user_count} users")
    logger.info(f"Target host: {environment.host}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Log test completion information"""
    logger.info("Load test completed")
    
    # Log final statistics
    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"Max response time: {stats.total.max_response_time}ms")
