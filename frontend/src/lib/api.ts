// API configuration
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export const API_ENDPOINTS = {
  CHAT: `${API_BASE_URL}/api/chat`,
  CHAT_STREAM: `${API_BASE_URL}/api/chat/stream`,
  HEALTH: `${API_BASE_URL}/health`,
  CATEGORIES: `${API_BASE_URL}/categories`,
};

interface StreamData {
  type: string;
  content?: string;
  conversation_id?: string;
  context_chunks?: unknown[];
  context_found?: boolean;
  model_used?: string;
  error?: string;
  is_final?: boolean;
}

class ApiService {
  async sendChatMessageStream(
    message: string,
    categories: string[] | null = null,
    conversationId: string | null = null,
    onMessage?: (data: StreamData) => void,
    onComplete?: (
      finalContent: string,
      finalMetadata: StreamData | null
    ) => void,
    onError?: (error: Error) => void
  ) {
    try {
      const payload = {
        message,
        categories,
        conversation_id: conversationId,
      };

      const response = await fetch(API_ENDPOINTS.CHAT_STREAM, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error("Failed to get response reader");
      }

      const decoder = new TextDecoder();
      let accumulatedResponse = "";
      let metadata: StreamData | null = null;
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();

        if (done) {
          onComplete?.(accumulatedResponse, metadata);
          break;
        }

        const chunk = decoder.decode(value, { stream: true });
        buffer += chunk;

        // Process complete lines from buffer
        const lines = buffer.split("\n");
        buffer = lines.pop() || ""; // Keep incomplete line in buffer

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));

              if (data.type === "metadata") {
                metadata = data;
                onMessage?.(data);
              } else if (data.type === "content") {
                if (data.content) {
                  accumulatedResponse += data.content;
                }
                onMessage?.(data);
              } else if (data.type === "error") {
                onError?.(new Error(data.error));
                return;
              }
            } catch (parseError) {
              console.error(
                "Error parsing SSE data:",
                parseError,
                "Line:",
                line
              );
            }
          }
        }
      }
    } catch (error) {
      console.error("Error in streaming chat:", error);
      onError?.(error as Error);
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
      console.error("Error checking health:", error);
      throw error;
    }
  }
}

const apiService = new ApiService();
export default apiService;
