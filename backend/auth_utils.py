"""
Authentication utilities for CAS integration
"""

import os
import httpx
import logging
from typing import Optional, Dict, Any
from urllib.parse import urlencode, quote
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class CASAuthenticator:
    def __init__(self):
        self.cas_server_url = os.getenv(
            "CAS_SERVER_URL", "https://login.iiit.ac.in/cas"
        )
        self.service_url = os.getenv("CAS_SERVICE_URL", "http://localhost:8000")
        self.admin_users = os.getenv("ADMIN_USERS", "").split(",")

    def get_login_url(self, service_url: Optional[str] = None) -> str:
        """Generate CAS login URL"""
        service = service_url or self.service_url
        return f"{self.cas_server_url}/login?service={quote(service)}"

    def get_logout_url(self, service_url: Optional[str] = None) -> str:
        """Generate CAS logout URL"""
        service = service_url or self.service_url
        return f"{self.cas_server_url}/logout?service={quote(service)}"

    async def validate_ticket(
        self, ticket: str, service_url: str
    ) -> Optional[Dict[str, Any]]:
        """Validate CAS ticket and return user information"""
        validation_url = f"{self.cas_server_url}/serviceValidate"

        params = {"ticket": ticket, "service": service_url}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(validation_url, params=params)
                response.raise_for_status()

                # Parse XML response
                user_info = self._parse_cas_response(response.text)

                if user_info:
                    # Check if user is admin
                    username = user_info.get("username", "")
                    email = f"{username}@iiit.ac.in"
                    user_info["email"] = email
                    user_info["is_admin"] = email in self.admin_users

                    logger.info(f"CAS validation successful for user: {username}")
                    return user_info
                else:
                    logger.warning("CAS validation failed: Invalid ticket")

        except Exception as e:
            logger.error(f"CAS validation error: {e}")

        return None

    def _parse_cas_response(self, xml_response: str) -> Optional[Dict[str, Any]]:
        """Parse CAS XML response and extract user information"""
        try:
            root = ET.fromstring(xml_response)

            # Define namespace
            ns = {"cas": "http://www.yale.edu/tp/cas"}

            # Check for authentication success
            success_elem = root.find(".//cas:authenticationSuccess", ns)
            if success_elem is not None:
                user_elem = success_elem.find("cas:user", ns)
                if user_elem is not None:
                    username = user_elem.text

                    # Extract additional attributes if available
                    attributes = {}
                    attrs_elem = success_elem.find("cas:attributes", ns)
                    if attrs_elem is not None:
                        for attr in attrs_elem:
                            attr_name = attr.tag.replace(
                                "{http://www.yale.edu/tp/cas}", ""
                            )
                            attributes[attr_name] = attr.text

                    return {
                        "username": username,
                        "full_name": attributes.get("displayName", username.title()),
                        "email": attributes.get("email", f"{username}@iiit.ac.in"),
                        "attributes": attributes,
                    }

            # Check for authentication failure
            failure_elem = root.find(".//cas:authenticationFailure", ns)
            if failure_elem is not None:
                logger.warning(f"CAS authentication failure: {failure_elem.text}")

        except ET.ParseError as e:
            logger.error(f"Error parsing CAS XML response: {e}")
        except Exception as e:
            logger.error(f"Unexpected error parsing CAS response: {e}")

        return None


# Session management utilities
class SessionManager:
    """Simple in-memory session management (use Redis/database in production)"""

    def __init__(self):
        self._sessions = {}

    def create_session(self, user_info: Dict[str, Any]) -> str:
        """Create a new session and return session ID"""
        import uuid

        session_id = str(uuid.uuid4())

        self._sessions[session_id] = {
            "user_info": user_info,
            "created_at": "now",  # Use proper datetime in production
            "last_accessed": "now",
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data by session ID"""
        session = self._sessions.get(session_id)
        if session:
            session["last_accessed"] = "now"  # Update access time
            return session["user_info"]
        return None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        return self._sessions.pop(session_id, None) is not None

    def cleanup_expired_sessions(self):
        """Remove expired sessions (implement proper expiry logic)"""
        # TODO: Implement session expiry based on timestamps
        pass


# Global session manager instance
session_manager = SessionManager()
