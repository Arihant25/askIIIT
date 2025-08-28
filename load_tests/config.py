"""
Base configuration for load testing askIIIT application.
Contains shared settings, utilities, and mock data.
"""

import os
import random
import string
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from faker import Faker
import json

# Initialize Faker for generating test data
fake = Faker()

@dataclass
class LoadTestConfig:
    """Configuration class for load testing parameters"""
    base_url: str = "http://localhost:8000"
    frontend_url: str = "http://localhost:3000"
    
    # Authentication settings
    test_users: List[str] = None
    admin_users: List[str] = None
    auth_token: str = "authenticated"  # Simple token for testing
    
    # Load testing parameters
    min_wait: int = 1000  # milliseconds
    max_wait: int = 3000  # milliseconds
    spawn_rate: int = 2   # users per second
    
    # Test data parameters
    max_query_length: int = 200
    max_message_length: int = 500
    
    def __post_init__(self):
        if self.test_users is None:
            self.test_users = [
                "testuser1@iiit.ac.in",
                "testuser2@iiit.ac.in", 
                "student@iiit.ac.in",
                "faculty@iiit.ac.in"
            ]
        
        if self.admin_users is None:
            self.admin_users = [
                "admin@iiit.ac.in",
                "arihant@iiit.ac.in"
            ]

# Global config instance
config = LoadTestConfig()

class TestDataGenerator:
    """Generates realistic test data for various endpoints"""
    
    def __init__(self):
        self.categories = ["faculty", "student", "hostel", "academics", "mess"]
        self.sample_queries = [
            "What are the hostel rules?",
            "How to apply for faculty position?",
            "What are the academic calendar dates?",
            "Mess menu for this week",
            "Student registration process",
            "Campus facilities available",
            "Admission requirements",
            "Fee structure details",
            "Library opening hours",
            "Sports complex facilities",
            "Transport services",
            "Health center information",
            "Course curriculum details",
            "Graduation requirements",
            "Scholarship information",
            "Placement statistics",
            "Research opportunities",
            "Faculty contact information",
            "Event calendar",
            "Examination schedule"
        ]
        
        self.conversation_starters = [
            "Hello, I need help with",
            "Can you tell me about",
            "I'm looking for information on",
            "Help me understand",
            "What do you know about",
            "Please explain",
            "I want to know more about",
            "Could you help me with"
        ]
        
        self.follow_up_questions = [
            "Can you provide more details?",
            "What else should I know?",
            "Are there any exceptions?",
            "How does this process work?",
            "What are the requirements?",
            "When is the deadline?",
            "Who should I contact?",
            "What documents are needed?"
        ]

    def generate_search_query(self) -> Dict[str, Any]:
        """Generate a realistic search query"""
        query = random.choice(self.sample_queries)
        
        # Sometimes add more specific details
        if random.random() < 0.3:
            query += f" for {fake.word()}"
        
        # Sometimes limit categories
        categories = None
        if random.random() < 0.4:
            categories = random.sample(self.categories, random.randint(1, 3))
        
        return {
            "query": query,
            "categories": categories,
            "limit": random.randint(5, 20)
        }

    def generate_chat_message(self, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a realistic chat message"""
        # Determine if this is a new conversation or continuation
        is_new_conversation = conversation_id is None
        
        if is_new_conversation:
            starter = random.choice(self.conversation_starters)
            topic = random.choice(self.sample_queries)
            message = f"{starter} {topic.lower()}"
        else:
            message = random.choice(self.follow_up_questions)
        
        # Sometimes add categories
        categories = None
        if random.random() < 0.3:
            categories = random.sample(self.categories, random.randint(1, 2))
        
        payload = {
            "message": message,
            "categories": categories
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
            
        return payload

    def generate_conversation_history(self, length: int = None) -> List[Dict[str, str]]:
        """Generate realistic conversation history"""
        if length is None:
            length = random.randint(2, 8)
        
        history = []
        for i in range(length):
            if i % 2 == 0:  # User message
                history.append({
                    "type": "user",
                    "content": random.choice(self.sample_queries)
                })
            else:  # Bot response
                history.append({
                    "type": "bot", 
                    "content": f"Based on the documents, {fake.text(max_nb_chars=200)}"
                })
        
        return history

    def generate_upload_metadata(self) -> Dict[str, str]:
        """Generate metadata for document upload testing"""
        return {
            "category": random.choice(self.categories),
            "description": fake.text(max_nb_chars=100)
        }

    def generate_user_info(self, is_admin: bool = False) -> Dict[str, Any]:
        """Generate user information for authentication"""
        if is_admin:
            email = random.choice(config.admin_users)
        else:
            email = random.choice(config.test_users)
        
        username = email.split('@')[0]
        
        return {
            "username": username,
            "email": email,
            "full_name": fake.name(),
            "is_admin": is_admin
        }

class MockAuthHeaders:
    """Helper for generating authentication headers"""
    
    @staticmethod
    def get_headers(is_admin: bool = False) -> Dict[str, str]:
        """Get mock authentication headers"""
        return {
            "Authorization": f"Bearer {config.auth_token}",
            "Content-Type": "application/json"
        }
    
    @staticmethod
    def get_user_info_headers() -> Dict[str, str]:
        """Get headers with user info for endpoints that need it"""
        return {
            "Authorization": f"Bearer {config.auth_token}",
            "Content-Type": "application/json",
            "X-User-Email": random.choice(config.test_users)
        }
    
    @staticmethod
    def get_admin_headers() -> Dict[str, str]:
        """Get admin headers for admin-only endpoints"""
        return {
            "Authorization": f"Bearer {config.auth_token}",
            "Content-Type": "application/json", 
            "X-User-Email": random.choice(config.admin_users),
            "X-User-Admin": "true"
        }

# Global test data generator
test_data = TestDataGenerator()

# Utility functions
def generate_random_string(length: int = 10) -> str:
    """Generate a random string of specified length"""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def get_random_category() -> str:
    """Get a random document category"""
    return random.choice(test_data.categories)

def get_random_categories(max_count: int = 3) -> List[str]:
    """Get a random list of categories"""
    count = random.randint(1, min(max_count, len(test_data.categories)))
    return random.sample(test_data.categories, count)

def simulate_typing_delay() -> float:
    """Simulate realistic typing delay for chat interactions"""
    return random.uniform(0.5, 2.0)

def load_environment_config() -> Dict[str, str]:
    """Load configuration from environment variables"""
    return {
        "base_url": os.getenv("LOAD_TEST_BASE_URL", config.base_url),
        "frontend_url": os.getenv("LOAD_TEST_FRONTEND_URL", config.frontend_url),
        "auth_token": os.getenv("LOAD_TEST_AUTH_TOKEN", config.auth_token),
        "admin_users": os.getenv("LOAD_TEST_ADMIN_USERS", ",".join(config.admin_users)).split(","),
        "test_users": os.getenv("LOAD_TEST_USERS", ",".join(config.test_users)).split(",")
    }

# Load any environment overrides
env_config = load_environment_config()
if env_config["base_url"] != config.base_url:
    config.base_url = env_config["base_url"]
if env_config["frontend_url"] != config.frontend_url:
    config.frontend_url = env_config["frontend_url"]
