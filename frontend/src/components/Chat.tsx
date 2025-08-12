"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  User,
  Bot,
  SendHorizontal,
  Book,
  School,
  Target,
  Microscope,
} from "lucide-react";
import ApiService from "@/lib/api";
import MarkdownRenderer from "./MarkdownRenderer";
import ReferencesSection from "./ReferencesSection";

const categories = [
  {
    id: "academics",
    label: "Academics",
    icon: Book,
    sources: ["academic-handbook"],
  },
  {
    id: "student",
    label: "Student",
    icon: School,
    sources: ["student-services"],
  },
  {
    id: "faculty",
    label: "Faculty",
    icon: Target,
    sources: ["faculty-handbook"],
  },
  {
    id: "hostel",
    label: "Hostel",
    icon: Microscope,
    sources: ["hostel-rules"],
  },
  { id: "mess", label: "Mess", icon: Microscope, sources: ["mess-menu"] },
];

interface Message {
  id: number;
  type: "user" | "bot";
  content: string;
  timestamp: Date;
  sources?: any[];
  groupedSources?: any[];
  processedContent?: string;
  isStreaming?: boolean;
  isError?: boolean;
  contextFound?: boolean;
}

interface ChatProps {
  chatStarted: boolean;
  setChatStarted: (started: boolean) => void;
}

const Chat: React.FC<ChatProps> = ({ chatStarted, setChatStarted }) => {
  const [inputValue, setInputValue] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedCategories, setSelectedCategories] = useState<string[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Function to process sources without adding citations to content
  const processMessageWithCitations = (content: string, sources: any[]) => {
    if (!sources || sources.length === 0) {
      return { processedContent: content, groupedSources: [] };
    }

    // Create a unique list of sources by filename
    const uniqueSources = new Map();
    sources.forEach((source) => {
      const key = source.filename;
      if (!uniqueSources.has(key)) {
        uniqueSources.set(key, {
          filename: source.filename,
          category: source.category,
        });
      }
    });

    const groupedSources = Array.from(uniqueSources.values());

    // Return the original content without any citations
    return { processedContent: content, groupedSources };
  };

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Don't focus if escape is pressed or if input is already focused
      if (e.key === "Escape" || document.activeElement === inputRef.current) {
        return;
      }

      // Don't focus if user is typing in another input/textarea or if modifier keys are pressed
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA" ||
        e.ctrlKey ||
        e.metaKey ||
        e.altKey
      ) {
        return;
      }

      // Focus the input for any other key press
      inputRef.current?.focus();
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now(),
      type: "user",
      content: inputValue.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputValue("");
    setIsLoading(true);
    if (!chatStarted) setChatStarted(true);

    const botMessageId = Date.now() + 1;
    const botMessage: Message = {
      id: botMessageId,
      type: "bot",
      content: "",
      timestamp: new Date(),
      sources: [],
      isStreaming: true,
    };
    setMessages((prev) => [...prev, botMessage]);

    try {
      let accumulatedContent = "";
      let metadata: any = null;

      // Prepare previous messages as context (excluding streaming/error messages)
      const contextMessages = messages
        .filter((msg) => !msg.isStreaming && !msg.isError)
        .map((msg) => ({
          type: msg.type,
          content: msg.content,
        }));

      await ApiService.sendChatMessageStream(
        userMessage.content,
        selectedCategories,
        conversationId,
        (data) => {
          if (data.type === "metadata") {
            metadata = data;
            if (!conversationId && data.conversation_id) {
              setConversationId(data.conversation_id);
            }
          } else if (data.type === "content") {
            accumulatedContent += data.content;
            setMessages((prev) =>
              prev.map((msg) => {
                if (msg.id === botMessageId) {
                  return {
                    ...msg,
                    content: accumulatedContent,
                    isStreaming: !data.is_final,
                  };
                }
                return msg;
              })
            );
          }
        },
        (finalContent, finalMetadata) => {
          const { processedContent, groupedSources } =
            processMessageWithCitations(
              finalContent || accumulatedContent,
              finalMetadata?.context_chunks || []
            );

          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === botMessageId
                ? {
                    ...msg,
                    content: finalContent || accumulatedContent,
                    processedContent: processedContent,
                    sources: finalMetadata?.context_chunks || [],
                    groupedSources: groupedSources,
                    isStreaming: false,
                    contextFound: finalMetadata?.context_found,
                  }
                : msg
            )
          );
        },
        (error) => {
          console.error("Chat streaming error:", error);
          const errorMessage: Message = {
            id: Date.now() + 1,
            type: "bot",
            content:
              "Sorry, I encountered an error while processing your question. Please try again.",
            timestamp: new Date(),
            isError: true,
          };
          setMessages((prev) => [...prev.slice(0, -1), errorMessage]);
        },
        contextMessages // Pass previous messages as context
      );
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: Message = {
        id: Date.now() + 1,
        type: "bot",
        content:
          "Sorry, I encountered an error while processing your question. Please try again.",
        timestamp: new Date(),
        isError: true,
      };
      setMessages((prev) => [...prev.slice(0, -1), errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {/* Messages area */}
      <div
        className={`relative z-20 flex-1 overflow-y-auto hide-scrollbar transition-all duration-1000 pb-36 ${
          chatStarted ? "opacity-100" : "opacity-0 pointer-events-none"
        }`}
      >
        <div className="max-w-4xl mx-auto px-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex items-end gap-4 mb-6 ${
                message.type === "user" ? "justify-end" : "justify-start"
              }`}
            >
              {message.type === "bot" && (
                <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-[#232946] flex items-center justify-center shadow-lg flex-shrink-0">
                  <Bot size={24} color="#93c5fd" />
                </div>
              )}

              <div
                className={`max-w-[80%] rounded-2xl px-4 md:px-6 py-3 md:py-4 text-sm md:text-base shadow-md break-words ${
                  message.type === "user"
                    ? "bg-[#60a5fa] text-[#181A20]"
                    : message.isError
                    ? "bg-red-500 text-white"
                    : "bg-[#93c5fd] text-[#232946]"
                }`}
              >
                {message.type === "bot" ? (
                  <MarkdownRenderer
                    content={message.processedContent || message.content}
                    className="message-content"
                  />
                ) : (
                  message.content
                )}

                {message.type === "bot" && message.isStreaming && (
                  <div className="typing-indicator ml-1 inline-flex items-center">
                    <span className="dot"></span>
                    <span className="dot"></span>
                    <span className="dot"></span>
                  </div>
                )}

                {message.type === "bot" &&
                  message.groupedSources &&
                  message.groupedSources.length > 0 && (
                    <ReferencesSection sources={message.groupedSources} />
                  )}
              </div>

              {message.type === "user" && (
                <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-[#232946] flex items-center justify-center shadow-lg flex-shrink-0">
                  <User size={24} color="#93c5fd" />
                </div>
              )}
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Form - Fixed at bottom */}
      <div className="fixed bottom-4 left-0 right-0 w-full z-30 px-4">
        <form
          className="max-w-4xl mx-auto bg-[#232946]/80 backdrop-blur-xl rounded-2xl p-4 shadow-lg"
          onSubmit={handleSubmit}
        >
          <div className="flex gap-3 mb-3">
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                selectedCategories.length > 0
                  ? `Ask about ${selectedCategories.join(", ")}...`
                  : `Ask me anything (all categories)...`
              }
              className="flex-1 px-4 md:px-5 py-3 md:py-3.5 rounded-xl border-none text-base md:text-lg bg-[#181A20] text-[#93c5fd] outline-none shadow-sm"
              disabled={isLoading}
            />
            <button
              type="submit"
              className={`px-6 md:px-7 rounded-xl bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] text-[#181A20] font-bold text-lg border-none cursor-pointer shadow-md transition-transform duration-200 flex items-center justify-center ${
                !inputValue.trim() || isLoading
                  ? "scale-100 cursor-not-allowed"
                  : "scale-105"
              }`}
              disabled={!inputValue.trim() || isLoading}
            >
              <SendHorizontal size={24} />
            </button>
          </div>

          <div className="flex gap-2 justify-center flex-wrap">
            {categories.map((cat) => {
              const IconComp = cat.icon;
              const isSelected = selectedCategories.includes(cat.id);
              return (
                <button
                  key={cat.id}
                  type="button"
                  onClick={() => {
                    setSelectedCategories((prev) => {
                      if (prev.includes(cat.id)) {
                        return prev.filter((id) => id !== cat.id);
                      } else {
                        return [...prev, cat.id];
                      }
                    });
                  }}
                  className={`flex items-center gap-2 rounded-lg px-3 md:px-4 py-2 font-medium text-sm md:text-base border-none cursor-pointer transition-all duration-200 ${
                    isSelected
                      ? "bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] text-[#181A20] font-bold shadow-md"
                      : "bg-[#181A20] text-[#93c5fd]"
                  }`}
                >
                  <IconComp
                    size={18}
                    color={isSelected ? "#181A20" : "#93c5fd"}
                  />
                  <span>{cat.label}</span>
                </button>
              );
            })}
          </div>
        </form>
      </div>
    </>
  );
};

export default Chat;
