import { API_ENDPOINTS } from '../config/api';

class ApiService {

  async sendChatMessage(message, categories = null, conversationId = null) {
    try {
      const payload = {
        message,
        categories,
        conversation_id: conversationId,
      };

      const response = await fetch(API_ENDPOINTS.CHAT, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Error sending chat message:', error);
      throw error;
    }
  }

  async sendChatMessageStream(message, categories = null, conversationId = null, onMessage, onComplete, onError) {
    try {
      const payload = {
        message,
        categories,
        conversation_id: conversationId,
      };

      const response = await fetch(API_ENDPOINTS.CHAT_STREAM, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedResponse = '';
      let metadata = null;

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          onComplete?.(accumulatedResponse, metadata);
          break;
        }

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === 'metadata') {
                metadata = data;
                onMessage?.(data);
              } else if (data.type === 'content') {
                if (data.content) {  // Only accumulate non-empty content
                  accumulatedResponse += data.content;
                }
                onMessage?.(data);
              } else if (data.type === 'error') {
                onError?.(new Error(data.error));
                return;
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      console.error('Error in streaming chat:', error);
      onError?.(error);
    }
  }

  async getHealth() {
    try {
      const response = await fetch(API_ENDPOINTS.HEALTH);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error checking health:', error);
      throw error;
    }
  }

  async getCategories() {
    try {
      const response = await fetch(API_ENDPOINTS.CATEGORIES);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('Error fetching categories:', error);
      throw error;
    }
  }
}

export default new ApiService();
