import { useState, useRef, useEffect } from 'react';
import { Book, School, Target, Microscope, User, Bot, ArrowUpRight } from 'lucide-react';
import ApiService from '../services/ApiService';

const categories = [
    { id: 'academics', label: 'Academics', icon: Book, sources: ['academic-handbook'] },
    { id: 'student', label: 'Student', icon: School, sources: ['student-services'] },
    { id: 'faculty', label: 'Faculty', icon: Target, sources: ['faculty-handbook'] },
    { id: 'hostel', label: 'Hostel', icon: Microscope, sources: ['hostel-rules'] },
    { id: 'mess', label: 'Mess', icon: Microscope, sources: ['mess-menu'] },
];

const ChatPage = () => {
    const [inputValue, setInputValue] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [selectedCategory, setSelectedCategory] = useState('academics');
    const [messages, setMessages] = useState([]);
    const [chatStarted, setChatStarted] = useState(false);
    const [conversationId, setConversationId] = useState(null);
    const [currentStreamingMessage, setCurrentStreamingMessage] = useState('');
    const [isStreaming, setIsStreaming] = useState(false);
    const [backendStatus, setBackendStatus] = useState('checking'); // 'checking', 'connected', 'disconnected'
    const messagesEndRef = useRef(null);

    // Check backend status on component mount
    useEffect(() => {
        const checkBackendStatus = async () => {
            try {
                await ApiService.getHealth();
                setBackendStatus('connected');
            } catch (error) {
                console.error('Backend health check failed:', error);
                setBackendStatus('disconnected');
            }
        };

        checkBackendStatus();
    }, []);

    useEffect(() => {
        if (messages.length === 0) {
            const selectedCatObj = categories.find(cat => cat.id === selectedCategory);
            setMessages([
                {
                    id: Date.now(),
                    type: 'bot',
                    content: `Hello! Select a category below and ask your question. I'll answer using official college documents.`,
                    timestamp: new Date(),
                    sources: selectedCatObj ? selectedCatObj.sources : []
                }
            ]);
        }
    }, [messages.length, selectedCategory]);

    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!inputValue.trim() || isLoading) return;

        const userMessage = {
            id: Date.now(),
            type: 'user',
            content: inputValue.trim(),
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsLoading(true);
        setIsStreaming(true);
        setCurrentStreamingMessage('');
        if (!chatStarted) setChatStarted(true);

        // Create placeholder bot message for streaming
        const botMessageId = Date.now() + 1;
        const botMessage = {
            id: botMessageId,
            type: 'bot',
            content: '',
            timestamp: new Date(),
            sources: [],
            isStreaming: true
        };
        setMessages(prev => [...prev, botMessage]);

        try {
            await ApiService.sendChatMessage(
                userMessage.content,
                selectedCategory,
                conversationId
            ).then(response => {
                // Update conversation ID if this is the first message
                if (!conversationId && response.conversation_id) {
                    setConversationId(response.conversation_id);
                }

                // Update the bot message with the complete response
                setMessages(prev =>
                    prev.map(msg =>
                        msg.id === botMessageId
                            ? {
                                ...msg,
                                content: response.response,
                                sources: response.context_chunks || [],
                                isStreaming: false,
                                contextFound: response.context_found
                            }
                            : msg
                    )
                );
            });
        } catch (error) {
            console.error('Chat error:', error);
            const errorMessage = {
                id: Date.now() + 1,
                type: 'bot',
                content: 'Sorry, I encountered an error while processing your question. Please try again.',
                timestamp: new Date(),
                isError: true
            };
            setMessages(prev => [...prev.slice(0, -1), errorMessage]);
        } finally {
            setIsLoading(false);
            setIsStreaming(false);
            setCurrentStreamingMessage('');
        }
    };

    return (
        <div className="chat-page min-h-screen w-screen bg-[#181A20] flex flex-col relative">
            {/* Title positioning changes after chat starts */}
            <div className={
                chatStarted
                    ? "absolute top-6 left-8 z-10 flex items-center gap-4"
                    : "flex-1 w-screen flex flex-col items-center justify-center absolute inset-0"
            }>
                <h1
                    className={
                        chatStarted
                            ? "text-[#60a5fa] font-semibold text-3xl text-left mb-0 tracking-wide"
                            : "text-[#60a5fa] font-semibold text-7xl text-center mb-[20vh] tracking-wide"
                    }
                >
                    askIIIT
                </h1>
                {/* Backend status indicator */}
                {chatStarted && (
                    <div className={`flex items-center gap-2 text-sm ${
                        backendStatus === 'connected' ? 'text-green-400' : 
                        backendStatus === 'disconnected' ? 'text-red-400' : 'text-yellow-400'
                    }`}>
                        <div className={`w-2 h-2 rounded-full ${
                            backendStatus === 'connected' ? 'bg-green-400' : 
                            backendStatus === 'disconnected' ? 'bg-red-400' : 'bg-yellow-400'
                        }`}></div>
                        {backendStatus === 'connected' ? 'Connected' : 
                         backendStatus === 'disconnected' ? 'Backend Offline' : 'Checking...'}
                    </div>
                )}
            </div>
            <div className={
                chatStarted
                    ? "flex flex-col mt-[5vh] h-[81vh] w-full p-4"
                    : "w-full max-w-xl mx-auto mt-8 flex flex-col items-center justify-end p-4"
            }>
                {/* Only show chat messages area after chatStarted */}
                {chatStarted && (
                    <div className="chat-messages w-full max-w-5xl flex-1 flex flex-col items-center mx-auto overflow-y-auto hide-scrollbar" >
                        {messages.map((message) => (
                            <div key={message.id} className={`message ${message.type} w-full flex items-end gap-4 ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                                {/* Avatar for bot/user */}
                                {message.type === 'bot' && (
                                    <div className="avatar-bot w-12 h-12 rounded-full bg-[#232946] flex items-center justify-center shadow-lg">
                                        <Bot size={32} color="#93c5fd" />
                                    </div>
                                )}
                                <div
                                    className={`max-w-[80%] rounded-2xl my-2 px-6 py-4 text-lg shadow-md whitespace-pre-line break-words ${message.type === 'user'
                                        ? 'bg-[#60a5fa] text-[#181A20]'
                                        : message.isError 
                                        ? 'bg-red-500 text-white'
                                        : 'bg-[#93c5fd] text-[#232946]'
                                        }`}
                                >
                                    {message.content}
                                    {/* Show sources if available */}
                                    {message.type === 'bot' && message.sources && message.sources.length > 0 && (
                                        <div className="mt-3 pt-3 border-t border-[#232946]/20">
                                            <div className="text-sm opacity-75 mb-2">Sources:</div>
                                            {message.sources.map((source, idx) => (
                                                <div key={idx} className="text-xs opacity-60 mb-1">
                                                    • {source.filename} ({source.category})
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                                {message.type === 'user' && (
                                    <div className="avatar-user w-12 h-12 rounded-full bg-[#232946] flex items-center justify-center shadow-lg">
                                        <User size={32} color="#93c5fd" />
                                    </div>
                                )}
                            </div>
                        ))}
                        {isLoading && (
                            <div className="message bot loading w-full flex justify-start">
                                <div className="max-w-[80%] bg-[#232946] text-[#93c5fd] rounded-2xl px-6 py-4 text-lg shadow-md">
                                    <div className="flex gap-1.5">
                                        <span className="w-2.5 h-2.5 bg-[#60a5fa] rounded-full inline-block animate-pulse"></span>
                                        <span className="w-2.5 h-2.5 bg-[#60a5fa] rounded-full inline-block animate-pulse delay-150"></span>
                                        <span className="w-2.5 h-2.5 bg-[#60a5fa] rounded-full inline-block animate-pulse delay-300"></span>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                )}
                <form
                    className={
                        chatStarted
                            ? "chat-input-form w-[calc(100%-theme(spacing.4))] max-w-5xl flex flex-col items-center justify-center bg-[#232946] rounded-2xl p-4 shadow-md mb-0 fixed bottom-4 left-1/2 -translate-x-1/2"
                            : "chat-input-form w-[calc(100%-theme(spacing.4))] max-w-3xl flex flex-col items-center justify-center bg-[#232946] rounded-2xl p-4 shadow-md mb-0 fixed bottom-8 left-1/2 -translate-x-1/2"
                    }
                    onSubmit={handleSubmit}
                >
                    <div className="w-full flex gap-3 mb-3">
                        <input
                            type="text"
                            value={inputValue}
                            onChange={(e) => setInputValue(e.target.value)}
                            placeholder={`Ask me anything...`}
                            className="chat-input flex-1 px-5 py-3.5 rounded-xl border-none text-lg bg-[#181A20] text-[#93c5fd] outline-none shadow-sm"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            className={`send-btn px-7 rounded-xl bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] text-[#181A20] font-bold text-lg border-none cursor-pointer shadow-md transition-transform duration-200 flex items-center justify-center ${!inputValue.trim() || isLoading ? 'scale-100' : 'scale-105'} ${isLoading ? 'cursor-not-allowed' : ''}`}
                            disabled={!inputValue.trim() || isLoading}
                        >
                            <ArrowUpRight size={28} style={{ color: '#181A20' }} />
                        </button>
                    </div>
                    <div className="w-full flex gap-2.5 justify-center mt-0">
                        {categories.map(cat => {
                            const IconComp = cat.icon;
                            return (
                                <button
                                    key={cat.id}
                                    type="button"
                                    onClick={() => setSelectedCategory(cat.id)}
                                    className={`flex items-center gap-2.5 rounded-lg px-4 py-2 font-medium text-base border-none cursor-pointer transition-all duration-200 ${selectedCategory === cat.id ? 'bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] text-[#181A20] font-bold shadow-md' : 'bg-[#181A20] text-[#93c5fd]'}`}
                                >
                                    <IconComp size={20} style={{ color: selectedCategory === cat.id ? '#181A20' : '#93c5fd' }} />
                                    <span>{cat.label}</span>
                                </button>
                            );
                        })}
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ChatPage;
