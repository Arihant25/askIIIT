"use client";

import { useState, useEffect } from "react";
import {
    Users,
    FileText,
    Database,
    LogOut,
    Upload,
    Trash2,
    RefreshCw,
    AlertCircle,
    CheckCircle,
    Activity,
    BarChart3,
    Terminal,
    Server,
    Eye,
    Play,
    Pause,
    X,
    Info,
    Hash,
    Calendar,
    User,
    Folder,
    Tag,
    Download,
    ExternalLink,
    Cpu,
    HardDrive,
    Clock,
    TrendingUp
} from "lucide-react";
import { AdminAPIService } from "@/services/adminAPI";

interface User {
    email: string;
    username: string;
    is_admin: boolean;
    full_name?: string;
    name?: string;
}

interface AdminDashboardProps {
    user: User;
}

interface SystemInfo {
    documents: {
        total: number;
        by_category: Record<string, number>;
    };
    chunks: {
        total: number;
    };
    models: {
        status: string;
    };
    database: {
        type: string;
        status: string;
        path: string;
    };
    processing: {
        is_indexing: boolean;
        current_document?: string;
        progress?: number;
        queue_size?: number;
        status_message?: string;
        processed_count?: number;
        total_count?: number;
    };
    backend: {
        status: string;
        uptime: string;
        memory_usage?: number;
        system_memory_percent?: number;
        cpu_percent?: number;
    };
}

interface Document {
    doc_id: string;
    name: string;
    category: string;
    description: string;
    created_at: string;
    author?: string;
    chunk_count?: number;
    embedding_count?: number;
    status?: string;
    file_size?: number;
    metadata?: Record<string, any>;
}

interface LogEntry {
    timestamp: string;
    level: string;
    message: string;
    source?: string;
}

export default function ModernAdminDashboard({ user }: AdminDashboardProps) {
    const [activeTab, setActiveTab] = useState("overview");
    const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
    const [documents, setDocuments] = useState<Document[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [uploadFiles, setUploadFiles] = useState<File[]>([]);
    const [uploading, setUploading] = useState(false);
    const [processing, setProcessing] = useState(false);
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
    const [showDocumentDetail, setShowDocumentDetail] = useState(false);
    const [autoRefresh, setAutoRefresh] = useState(true);
    const [logFilter, setLogFilter] = useState<string>("");
    const [logLevel, setLogLevel] = useState<string>("");

    useEffect(() => {
        loadSystemInfo();
        loadDocuments();
        loadLogs();
        
        if (autoRefresh) {
            const interval = setInterval(() => {
                loadSystemInfo();
                loadLogs();
            }, 5000);
            return () => clearInterval(interval);
        }
    }, [autoRefresh]);

    // Separate effect for log filtering
    useEffect(() => {
        loadLogs();
    }, [logFilter, logLevel]);

    const loadSystemInfo = async () => {
        try {
            const data = await AdminAPIService.getSystemInfo();
            setSystemInfo(data);
        } catch (err) {
            setError("Failed to load system information");
        }
    };

    const loadDocuments = async () => {
        try {
            const data = await AdminAPIService.getDocuments();
            setDocuments(data.documents || []);
        } catch (err) {
            setError("Failed to load documents");
        } finally {
            setLoading(false);
        }
    };

    const loadLogs = async () => {
        try {
            const params = new URLSearchParams();
            params.append('lines', '500');
            
            // Add level filter if specified
            if (logLevel && logLevel.trim() !== '') {
                params.append('level', logLevel.trim());
            }
            
            console.log('Loading logs with params:', params.toString());
            
            const response = await fetch(`/api/admin/logs?${params.toString()}`, {
                credentials: 'include',
            });
            
            if (response.ok) {
                const data = await response.json();
                let filteredLogs = data.logs || [];
                
                console.log('Received logs:', filteredLogs.length, 'entries');
                
                // Apply client-side text filter if specified
                if (logFilter && logFilter.trim() !== '') {
                    const filterText = logFilter.toLowerCase().trim();
                    filteredLogs = filteredLogs.filter((log: LogEntry) => {
                        const message = (log.message || '').toLowerCase();
                        const source = (log.source || '').toLowerCase();
                        return message.includes(filterText) || source.includes(filterText);
                    });
                    console.log('After text filtering:', filteredLogs.length, 'entries');
                }
                
                setLogs(filteredLogs);
            } else {
                console.error('Failed to load logs:', response.status, response.statusText);
                setLogs([]);
            }
        } catch (err) {
            console.error('Error loading logs:', err);
            setLogs([]);
        }
    };

    const handleMultipleUpload = async () => {
        if (uploadFiles.length === 0) return;

        setUploading(true);
        try {
            for (const file of uploadFiles) {
                const formData = new FormData();
                formData.append("file", file);
                await AdminAPIService.uploadDocument(formData);
            }

            setUploadFiles([]);
            loadDocuments();
            loadSystemInfo();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Upload failed");
        } finally {
            setUploading(false);
        }
    };

    const handleProcessUploaded = async () => {
        setProcessing(true);
        try {
            await AdminAPIService.startBulkProcessing();
            loadSystemInfo();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Processing failed");
        } finally {
            setProcessing(false);
        }
    };

    const handleDownloadDocument = async (docId: string, filename: string) => {
        try {
            const blob = await AdminAPIService.downloadDocument(docId);
            const url = window.URL.createObjectURL(blob);
            window.open(url, '_blank');
            window.URL.revokeObjectURL(url);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Download failed");
        }
    };

    const handleDeleteDocument = async (docId: string) => {
        if (!confirm("Are you sure you want to delete this document?")) return;

        try {
            await AdminAPIService.deleteDocument(docId);
            loadDocuments();
            loadSystemInfo();
        } catch (err) {
            setError("Failed to delete document");
        }
    };

    const handleReindex = async () => {
        if (!confirm("This will reindex all documents. Continue?")) return;

        try {
            await AdminAPIService.reindexDocuments();
            loadSystemInfo();
            setError("");
        } catch (err) {
            setError("Reindexing failed");
        }
    };

    const tabs = [
        { id: "overview", label: "Overview", icon: BarChart3 },
        { id: "documents", label: "Documents", icon: FileText },
        { id: "logs", label: "Logs", icon: Terminal },
        { id: "users", label: "Users", icon: Users },
    ];

    const categories = [
        { value: "faculty", label: "Faculty Data" },
        { value: "student", label: "Student Data" },
        { value: "hostel", label: "Hostels" },
        { value: "academics", label: "Academics" },
        { value: "mess", label: "Messes" },
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-[#0f1419] via-[#181A20] to-[#232946] relative">
            {/* Animated Background Elements */}
            <div className="fixed inset-0 overflow-hidden pointer-events-none">
                <div className="absolute -top-32 -left-32 w-64 h-64 bg-gradient-to-r from-[#60a5fa]/10 to-[#93c5fd]/10 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute -bottom-32 -right-32 w-96 h-96 bg-gradient-to-r from-[#6699ee]/10 to-[#60a5fa]/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
            </div>

            {/* Modern Header */}
            <header className="relative bg-black/30 backdrop-blur-2xl border-b border-white/10 sticky top-0 z-50">
                <div className="absolute inset-0 bg-gradient-to-r from-[#60a5fa]/5 to-[#93c5fd]/5"></div>
                <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-20">
                        <div className="flex items-center space-x-6">
                            <div className="relative">
                                <div className="w-14 h-14 bg-gradient-to-br from-[#60a5fa] via-[#93c5fd] to-[#6699ee] rounded-2xl flex items-center justify-center shadow-2xl shadow-blue-500/25">
                                    <Server className="w-7 h-7 text-white" />
                                </div>
                                <div className="absolute -inset-1 bg-gradient-to-br from-[#60a5fa] to-[#93c5fd] rounded-2xl blur opacity-30 animate-pulse"></div>
                            </div>
                            <div>
                                <h1 className="text-3xl font-bold bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#6699ee] bg-clip-text text-transparent">
                                    Admin Console
                                </h1>
                                <p className="text-gray-400 text-sm font-medium">System Management Dashboard</p>
                            </div>
                        </div>
                        <div className="flex items-center gap-8">
                            <div className="flex items-center gap-4">
                                <label className="flex items-center gap-2 text-sm text-gray-300">
                                    <input
                                        type="checkbox"
                                        checked={autoRefresh}
                                        onChange={(e) => setAutoRefresh(e.target.checked)}
                                        className="rounded bg-white/10 border-white/20 text-blue-500 focus:ring-blue-500/25 focus:ring-offset-0"
                                    />
                                    Auto-refresh
                                </label>
                                <button
                                    onClick={() => {
                                        loadSystemInfo();
                                        loadDocuments();
                                        loadLogs();
                                    }}
                                    className="flex items-center gap-2 px-4 py-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-400 hover:text-blue-300 rounded-xl transition-all duration-200 backdrop-blur-sm border border-blue-500/20"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Refresh
                                </button>
                            </div>
                            <div className="text-right">
                                <p className="text-white font-semibold">{user.full_name || user.name || user.username}</p>
                                <p className="text-gray-400 text-sm">Administrator</p>
                            </div>
                            <button
                                onClick={() => window.location.href = "/api/auth/logout"}
                                className="flex items-center gap-2 px-4 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 hover:text-red-300 rounded-xl transition-all duration-200 backdrop-blur-sm border border-red-500/20"
                            >
                                <LogOut className="w-4 h-4" />
                                Logout
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-12">
                {/* Error Alert - Floating */}
                {error && (
                    <div className="fixed top-24 right-6 z-50 animate-in slide-in-from-right-5 duration-300">
                        <div className="p-6 bg-red-500/10 backdrop-blur-2xl border border-red-500/30 rounded-2xl flex items-center gap-4 shadow-2xl max-w-md">
                            <div className="w-12 h-12 bg-red-500/20 rounded-full flex items-center justify-center flex-shrink-0">
                                <AlertCircle className="w-6 h-6 text-red-400" />
                            </div>
                            <div className="flex-1">
                                <p className="text-red-300 font-medium">{error}</p>
                                <p className="text-red-400/70 text-sm mt-1">Click to dismiss</p>
                            </div>
                            <button
                                onClick={() => setError("")}
                                className="w-8 h-8 bg-red-500/20 hover:bg-red-500/30 rounded-full flex items-center justify-center transition-colors flex-shrink-0"
                            >
                                <X className="w-4 h-4 text-red-400" />
                            </button>
                        </div>
                    </div>
                )}

                {/* Modern Navigation Tabs */}
                <div className="flex items-center justify-center">
                    <nav className="flex bg-black/30 backdrop-blur-2xl p-2 rounded-3xl border border-white/10 shadow-2xl">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`relative flex items-center gap-3 px-8 py-4 text-sm font-semibold rounded-2xl transition-all duration-300 group ${
                                    activeTab === tab.id
                                        ? "bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] text-white shadow-xl shadow-blue-500/25"
                                        : "text-gray-400 hover:text-white hover:bg-white/5"
                                }`}
                            >
                                <tab.icon className={`w-5 h-5 transition-all duration-300 ${
                                    activeTab === tab.id ? 'scale-110' : 'group-hover:scale-105'
                                }`} />
                                {tab.label}
                                {activeTab === tab.id && (
                                    <div className="absolute inset-0 bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] rounded-2xl blur-lg opacity-50 -z-10 animate-pulse" />
                                )}
                            </button>
                        ))}
                    </nav>
                </div>

                {/* Tab Content */}
                {activeTab === "overview" && (
                    <div className="space-y-12 animate-in fade-in-0 duration-500">
                        <div className="text-center">
                            <h2 className="text-4xl font-bold bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#6699ee] bg-clip-text text-transparent mb-4">
                                System Overview
                            </h2>
                            <p className="text-gray-400 text-lg">Real-time monitoring and system statistics</p>
                        </div>

                        {systemInfo && (
                            <>
                                {/* Main Stats Cards */}
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                                    <div className="group relative bg-black/20 backdrop-blur-xl p-6 rounded-3xl border border-white/10 hover:border-blue-500/30 transition-all duration-300 hover:scale-105">
                                        <div className="absolute inset-0 bg-gradient-to-br from-blue-500/5 to-cyan-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                        <div className="relative flex items-center gap-4">
                                            <div className="w-16 h-16 bg-gradient-to-br from-blue-500/20 to-blue-600/20 rounded-2xl flex items-center justify-center">
                                                <FileText className="w-8 h-8 text-blue-400" />
                                            </div>
                                            <div>
                                                <p className="text-gray-400 text-sm font-medium">Documents</p>
                                                <p className="text-3xl font-bold text-white">{systemInfo.documents.total}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="group relative bg-black/20 backdrop-blur-xl p-6 rounded-3xl border border-white/10 hover:border-green-500/30 transition-all duration-300 hover:scale-105">
                                        <div className="absolute inset-0 bg-gradient-to-br from-green-500/5 to-emerald-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                        <div className="relative flex items-center gap-4">
                                            <div className="w-16 h-16 bg-gradient-to-br from-green-500/20 to-green-600/20 rounded-2xl flex items-center justify-center">
                                                <Database className="w-8 h-8 text-green-400" />
                                            </div>
                                            <div>
                                                <p className="text-gray-400 text-sm font-medium">Chunks</p>
                                                <p className="text-3xl font-bold text-white">{systemInfo.chunks.total}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="group relative bg-black/20 backdrop-blur-xl p-6 rounded-3xl border border-white/10 hover:border-purple-500/30 transition-all duration-300 hover:scale-105">
                                        <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-violet-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                        <div className="relative flex items-center gap-4">
                                            <div className="w-16 h-16 bg-gradient-to-br from-purple-500/20 to-purple-600/20 rounded-2xl flex items-center justify-center">
                                                <Activity className={`w-8 h-8 ${systemInfo.models.status === 'healthy' ? 'text-green-400' : 'text-red-400'}`} />
                                            </div>
                                            <div>
                                                <p className="text-gray-400 text-sm font-medium">AI Models</p>
                                                <p className="text-lg font-bold text-white capitalize">{systemInfo.models.status}</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="group relative bg-black/20 backdrop-blur-xl p-6 rounded-3xl border border-white/10 hover:border-orange-500/30 transition-all duration-300 hover:scale-105">
                                        <div className="absolute inset-0 bg-gradient-to-br from-orange-500/5 to-red-500/5 rounded-3xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                                        <div className="relative flex items-center gap-4">
                                            <div className="w-16 h-16 bg-gradient-to-br from-orange-500/20 to-orange-600/20 rounded-2xl flex items-center justify-center">
                                                <Server className={`w-8 h-8 ${systemInfo.backend.status === 'healthy' ? 'text-green-400' : 'text-red-400'}`} />
                                            </div>
                                            <div>
                                                <p className="text-gray-400 text-sm font-medium">Backend</p>
                                                <p className="text-lg font-bold text-white">{systemInfo.backend.uptime}</p>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* System Performance */}
                                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                                    <div className="bg-black/20 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                                        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                                            <TrendingUp className="w-6 h-6 text-blue-400" />
                                            System Performance
                                        </h3>
                                        <div className="space-y-6">
                                            <div className="space-y-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <Cpu className="w-5 h-5 text-blue-400" />
                                                        <span className="text-gray-300">CPU Usage ({systemInfo.backend.cpu_count || 0} cores)</span>
                                                    </div>
                                                    <span className="text-white font-semibold">{systemInfo.backend.cpu_percent || 0}%</span>
                                                </div>
                                                <div className="w-full bg-gray-700 rounded-full h-2">
                                                    <div 
                                                        className="bg-blue-500 h-2 rounded-full transition-all duration-300" 
                                                        style={{ width: `${Math.min(systemInfo.backend.cpu_percent || 0, 100)}%` }}
                                                    ></div>
                                                </div>
                                            </div>
                                            
                                            <div className="space-y-4">
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <HardDrive className="w-5 h-5 text-green-400" />
                                                        <span className="text-gray-300">System Memory</span>
                                                    </div>
                                                    <span className="text-white font-semibold">
                                                        {systemInfo.backend.system_memory_used || 0}GB / {systemInfo.backend.system_memory_total || 0}GB
                                                    </span>
                                                </div>
                                                <div className="w-full bg-gray-700 rounded-full h-2">
                                                    <div 
                                                        className="bg-green-500 h-2 rounded-full transition-all duration-300" 
                                                        style={{ width: `${Math.min(systemInfo.backend.system_memory_percent || 0, 100)}%` }}
                                                    ></div>
                                                </div>
                                                <div className="text-sm text-gray-400">
                                                    Process: {systemInfo.backend.memory_usage || 0}MB ({systemInfo.backend.memory_percent || 0}%)
                                                </div>
                                            </div>
                                            
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-3">
                                                    <Clock className="w-5 h-5 text-purple-400" />
                                                    <span className="text-gray-300">Uptime</span>
                                                </div>
                                                <span className="text-white font-semibold">{systemInfo.backend.uptime}</span>
                                            </div>
                                            
                                            {systemInfo.backend.disk_usage_percent && (
                                                <div className="space-y-2">
                                                    <div className="flex items-center justify-between">
                                                        <div className="flex items-center gap-3">
                                                            <Database className="w-5 h-5 text-yellow-400" />
                                                            <span className="text-gray-300">Disk Usage</span>
                                                        </div>
                                                        <span className="text-white font-semibold">{systemInfo.backend.disk_usage_percent}%</span>
                                                    </div>
                                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                                        <div 
                                                            className="bg-yellow-500 h-2 rounded-full transition-all duration-300" 
                                                            style={{ width: `${Math.min(systemInfo.backend.disk_usage_percent, 100)}%` }}
                                                        ></div>
                                                    </div>
                                                </div>
                                            )}
                                            
                                            {systemInfo.backend.load_average && (
                                                <div className="flex items-center justify-between">
                                                    <div className="flex items-center gap-3">
                                                        <TrendingUp className="w-5 h-5 text-orange-400" />
                                                        <span className="text-gray-300">Load Average</span>
                                                    </div>
                                                    <span className="text-white font-semibold">{systemInfo.backend.load_average}</span>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    <div className="bg-black/20 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                                        <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                                            <Server className="w-6 h-6 text-purple-400" />
                                            Processing Status
                                        </h3>
                                        <div className="space-y-4">
                                            <div className="flex items-center gap-3">
                                                <div className={`w-3 h-3 rounded-full ${systemInfo.processing.is_indexing ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
                                                <span className="text-gray-300">
                                                    {systemInfo.processing.is_indexing ? 'Processing Documents' : 'Idle'}
                                                </span>
                                            </div>
                                            {systemInfo.processing.current_document && (
                                                <div>
                                                    <p className="text-gray-400 text-sm">Current Document</p>
                                                    <p className="text-white font-medium truncate">{systemInfo.processing.current_document}</p>
                                                </div>
                                            )}
                                            <div className="flex justify-between text-sm">
                                                <span className="text-gray-400">Queue Size</span>
                                                <span className="text-white">{systemInfo.processing.queue_size || 0}</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>

                                {/* Document Categories */}
                                <div className="bg-black/20 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                                    <h3 className="text-2xl font-bold text-white mb-8 text-center">Documents by Category</h3>
                                    <div className="grid grid-cols-2 md:grid-cols-5 gap-6">
                                        {Object.entries(systemInfo.documents.by_category).map(([category, count]) => (
                                            <div key={category} className="text-center group">
                                                <div className="w-20 h-20 mx-auto bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-2xl flex items-center justify-center mb-3 group-hover:scale-110 transition-transform duration-200">
                                                    <span className="text-3xl font-bold text-white">{count}</span>
                                                </div>
                                                <p className="text-gray-400 capitalize font-medium">{category}</p>
                                            </div>
                                        ))}
                                    </div>
                                </div>

                                {/* Quick Actions */}
                                <div className="bg-black/20 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                                    <h3 className="text-xl font-bold text-white mb-6">Quick Actions</h3>
                                    <div className="flex flex-wrap gap-4">
                                        <button
                                            onClick={handleReindex}
                                            className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 hover:from-blue-500/30 hover:to-cyan-500/30 text-blue-400 hover:text-blue-300 rounded-2xl transition-all duration-200 border border-blue-500/20"
                                        >
                                            <RefreshCw className="w-5 h-5" />
                                            Reindex Documents
                                        </button>
                                        <button
                                            onClick={handleProcessUploaded}
                                            disabled={processing}
                                            className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 hover:from-green-500/30 hover:to-emerald-500/30 text-green-400 hover:text-green-300 rounded-2xl transition-all duration-200 border border-green-500/20 disabled:opacity-50"
                                        >
                                            {processing ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                                            {processing ? 'Processing...' : 'Start Bulk Processing'}
                                        </button>
                                    </div>
                                </div>
                            </>
                        )}
                    </div>
                )}

                {/* Logs Tab - Full Width */}
                {activeTab === "logs" && (
                    <div className="space-y-8 animate-in fade-in-0 duration-500">
                        <div className="text-center">
                            <h2 className="text-4xl font-bold bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#6699ee] bg-clip-text text-transparent mb-4">
                                System Logs
                            </h2>
                            <p className="text-gray-400 text-lg">Real-time system logs and monitoring • {logs.length} entries</p>
                        </div>

                        {/* Log Filters & Controls */}
                        <div className="bg-black/20 backdrop-blur-xl p-6 rounded-3xl border border-white/10">
                            <div className="flex flex-wrap items-center justify-between gap-6">
                                <div className="flex flex-wrap items-center gap-4">
                                    <div className="flex items-center gap-2">
                                        <label className="text-sm text-gray-400 font-medium">Search:</label>
                                        <input
                                            type="text"
                                            placeholder="Filter logs..."
                                            value={logFilter}
                                            onChange={(e) => setLogFilter(e.target.value)}
                                            className="px-4 py-2 bg-white/5 backdrop-blur border border-white/10 rounded-xl text-white text-sm focus:ring-2 focus:ring-blue-500/50 focus:border-transparent placeholder-gray-500"
                                        />
                                    </div>
                                    <div className="flex items-center gap-2">
                                        <label className="text-sm text-gray-400 font-medium">Level:</label>
                                        <select
                                            value={logLevel}
                                            onChange={(e) => setLogLevel(e.target.value)}
                                            className="px-4 py-2 bg-white/5 backdrop-blur border border-white/10 rounded-xl text-white text-sm focus:ring-2 focus:ring-blue-500/50 focus:border-transparent"
                                        >
                                            <option value="">All Levels</option>
                                            <option value="DEBUG">DEBUG</option>
                                            <option value="INFO">INFO</option>
                                            <option value="WARNING">WARNING</option>
                                            <option value="ERROR">ERROR</option>
                                            <option value="CRITICAL">CRITICAL</option>
                                        </select>
                                    </div>
                                </div>
                                <div className="flex items-center gap-4">
                                    <button
                                        onClick={() => {
                                            setLogFilter("");
                                            setLogLevel("");
                                        }}
                                        className="px-4 py-2 bg-gray-500/20 hover:bg-gray-500/30 text-gray-300 hover:text-white rounded-xl transition-all duration-200"
                                    >
                                        Clear Filters
                                    </button>
                                    <button
                                        onClick={loadLogs}
                                        className="flex items-center gap-2 px-6 py-2 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 hover:from-blue-500/30 hover:to-cyan-500/30 text-blue-400 hover:text-blue-300 rounded-xl transition-all duration-200 border border-blue-500/20"
                                    >
                                        <RefreshCw className="w-4 h-4" />
                                        Refresh
                                    </button>
                                </div>
                            </div>
                        </div>

                        {/* Full-Width Logs Display */}
                        <div className="bg-black/30 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden">
                            <div className="h-[80vh] overflow-y-auto">
                                <div className="p-6 space-y-1 font-mono text-sm">
                                    {logs.length === 0 ? (
                                        <div className="flex items-center justify-center h-full min-h-[400px]">
                                            <div className="text-center">
                                                <Terminal className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                                                <p className="text-gray-400 text-lg">No logs available</p>
                                                {(logFilter || logLevel) && (
                                                    <p className="text-gray-500 text-sm mt-2">Try adjusting your filters</p>
                                                )}
                                            </div>
                                        </div>
                                    ) : (
                                        logs.map((log, index) => (
                                            <div key={index} className="flex gap-4 py-3 px-4 hover:bg-white/5 rounded-xl transition-colors duration-150 group">
                                                <span className="text-gray-400 text-xs whitespace-nowrap min-w-[100px] font-medium">
                                                    {new Date(log.timestamp).toLocaleTimeString()}
                                                </span>
                                                <span className={`text-xs font-bold whitespace-nowrap min-w-[80px] px-2 py-1 rounded-lg ${
                                                    log.level === 'ERROR' || log.level === 'CRITICAL' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
                                                    log.level === 'WARN' || log.level === 'WARNING' ? 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/30' :
                                                    log.level === 'INFO' ? 'bg-blue-500/20 text-blue-300 border border-blue-500/30' :
                                                    log.level === 'DEBUG' ? 'bg-gray-500/20 text-gray-300 border border-gray-500/30' :
                                                    'bg-gray-500/20 text-gray-300 border border-gray-500/30'
                                                }`}>
                                                    {log.level}
                                                </span>
                                                <span className="text-purple-400 text-xs whitespace-nowrap min-w-[120px] px-2 py-1 bg-purple-500/10 rounded-lg border border-purple-500/20">
                                                    [{log.source || 'system'}]
                                                </span>
                                                <span className="text-gray-100 text-sm flex-1 leading-relaxed group-hover:text-white transition-colors duration-150">
                                                    {log.message}
                                                </span>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </div>

                        {/* Status Footer */}
                        <div className="flex justify-between items-center text-sm text-gray-400 bg-black/20 backdrop-blur-xl p-4 rounded-2xl border border-white/10">
                            <div>
                                Auto-refresh: <span className={autoRefresh ? 'text-green-400' : 'text-gray-400'}>{autoRefresh ? 'On' : 'Off'}</span>
                                {" • "}
                                Last updated: {new Date().toLocaleTimeString()}
                            </div>
                            <div>
                                Showing {logs.length} of latest entries
                            </div>
                        </div>
                    </div>
                )}

                {/* Documents Tab */}
                {activeTab === "documents" && (
                    <div className="space-y-8 animate-in fade-in-0 duration-500">
                        <div className="text-center">
                            <h2 className="text-4xl font-bold bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#6699ee] bg-clip-text text-transparent mb-4">
                                Document Management
                            </h2>
                            <p className="text-gray-400 text-lg">Upload, manage, and organize your knowledge base</p>
                        </div>

                        {/* Upload Section */}
                        <div className="bg-gradient-to-br from-black/30 via-black/20 to-black/30 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                                <Upload className="w-6 h-6 text-blue-400" />
                                Upload Documents
                            </h3>
                            <div className="space-y-6">
                                <div className="relative">
                                    <input
                                        id="uploadFileInput"
                                        type="file"
                                        accept=".pdf"
                                        multiple
                                        onChange={(e) => setUploadFiles(Array.from(e.target.files || []))}
                                        className="block w-full text-sm text-gray-400 file:mr-4 file:py-3 file:px-6 file:rounded-2xl file:border-0 file:bg-gradient-to-r file:from-blue-500/20 file:to-cyan-500/20 file:text-blue-300 file:font-semibold hover:file:from-blue-500/30 hover:file:to-cyan-500/30 file:cursor-pointer cursor-pointer file:transition-all file:duration-200 file:border file:border-blue-500/30"
                                    />
                                </div>
                                {uploadFiles.length > 0 && (
                                    <div className="space-y-4">
                                        <p className="text-gray-300 font-medium">Selected files ({uploadFiles.length}):</p>
                                        <div className="max-h-40 overflow-y-auto space-y-2 bg-black/30 rounded-2xl p-4 border border-white/10">
                                            {uploadFiles.map((file, index) => (
                                                <div key={index} className="flex items-center justify-between bg-white/5 p-3 rounded-xl hover:bg-white/10 transition-colors">
                                                    <span className="text-white font-medium truncate flex-1">{file.name}</span>
                                                    <button
                                                        onClick={() => setUploadFiles(files => files.filter((_, i) => i !== index))}
                                                        className="w-8 h-8 bg-red-500/20 hover:bg-red-500/30 text-red-400 hover:text-red-300 rounded-full flex items-center justify-center transition-colors ml-3 flex-shrink-0"
                                                    >
                                                        <X className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="flex gap-4">
                                            <button
                                                onClick={handleMultipleUpload}
                                                disabled={uploading}
                                                className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-blue-500/20 to-cyan-500/20 hover:from-blue-500/30 hover:to-cyan-500/30 disabled:opacity-50 text-blue-300 hover:text-blue-200 rounded-2xl transition-all duration-200 border border-blue-500/30"
                                            >
                                                <Upload className="w-5 h-5" />
                                                {uploading ? 'Uploading...' : 'Upload Files'}
                                            </button>
                                            <button
                                                onClick={handleProcessUploaded}
                                                disabled={processing || uploadFiles.length === 0}
                                                className="flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 hover:from-green-500/30 hover:to-emerald-500/30 disabled:opacity-50 text-green-300 hover:text-green-200 rounded-2xl transition-all duration-200 border border-green-500/30"
                                            >
                                                <Play className="w-5 h-5" />
                                                {processing ? 'Processing...' : 'Process Documents'}
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Documents Table */}
                        <div className="bg-black/20 backdrop-blur-xl rounded-3xl border border-white/10 overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead className="bg-black/40 backdrop-blur border-b border-white/10">
                                        <tr>
                                            <th className="px-8 py-6 text-left text-sm font-semibold text-gray-300 uppercase tracking-wider">
                                                Document
                                            </th>
                                            <th className="px-8 py-6 text-left text-sm font-semibold text-gray-300 uppercase tracking-wider">
                                                Category
                                            </th>
                                            <th className="px-8 py-6 text-left text-sm font-semibold text-gray-300 uppercase tracking-wider">
                                                Chunks
                                            </th>
                                            <th className="px-8 py-6 text-left text-sm font-semibold text-gray-300 uppercase tracking-wider">
                                                Status
                                            </th>
                                            <th className="px-8 py-6 text-left text-sm font-semibold text-gray-300 uppercase tracking-wider">
                                                Created
                                            </th>
                                            <th className="px-8 py-6 text-left text-sm font-semibold text-gray-300 uppercase tracking-wider">
                                                Actions
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-white/10">
                                        {documents.map((doc) => (
                                            <tr 
                                                key={doc.doc_id} 
                                                className="hover:bg-white/5 cursor-pointer transition-colors duration-200 group"
                                                onClick={() => {
                                                    setSelectedDocument(doc);
                                                    setShowDocumentDetail(true);
                                                }}
                                            >
                                                <td className="px-8 py-6">
                                                    <div className="flex items-center gap-3">
                                                        <div className="w-10 h-10 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-xl flex items-center justify-center">
                                                            <FileText className="w-5 h-5 text-blue-400" />
                                                        </div>
                                                        <div>
                                                            <p className="text-white font-medium truncate max-w-xs group-hover:text-blue-300 transition-colors">{doc.name}</p>
                                                            <p className="text-gray-400 text-sm truncate max-w-xs">{doc.description}</p>
                                                        </div>
                                                    </div>
                                                </td>
                                                <td className="px-8 py-6">
                                                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${
                                                        doc.category === 'academics' ? 'bg-blue-500/20 text-blue-300 border-blue-500/30' :
                                                        doc.category === 'student' ? 'bg-green-500/20 text-green-300 border-green-500/30' :
                                                        doc.category === 'faculty' ? 'bg-purple-500/20 text-purple-300 border-purple-500/30' :
                                                        doc.category === 'hostel' ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' :
                                                        doc.category === 'mess' ? 'bg-red-500/20 text-red-300 border-red-500/30' :
                                                        'bg-gray-500/20 text-gray-300 border-gray-500/30'
                                                    }`}>
                                                        {doc.category}
                                                    </span>
                                                </td>
                                                <td className="px-8 py-6">
                                                    <div className="flex items-center gap-2">
                                                        <Hash className="w-4 h-4 text-gray-400" />
                                                        <span className="text-white font-medium">{doc.chunk_count || 0}</span>
                                                    </div>
                                                </td>
                                                <td className="px-8 py-6">
                                                    <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold border ${
                                                        doc.status === 'processed' ? 'bg-green-500/20 text-green-300 border-green-500/30' :
                                                        doc.status === 'processing' ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' :
                                                        doc.status === 'failed' ? 'bg-red-500/20 text-red-300 border-red-500/30' :
                                                        'bg-gray-500/20 text-gray-300 border-gray-500/30'
                                                    }`}>
                                                        {doc.status || 'processed'}
                                                    </span>
                                                </td>
                                                <td className="px-8 py-6">
                                                    <div className="flex items-center gap-2 text-gray-400">
                                                        <Calendar className="w-4 h-4" />
                                                        <span className="text-sm">{new Date(doc.created_at).toLocaleDateString()}</span>
                                                    </div>
                                                </td>
                                                <td className="px-8 py-6">
                                                    <div className="flex items-center gap-2">
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setSelectedDocument(doc);
                                                                setShowDocumentDetail(true);
                                                            }}
                                                            className="w-10 h-10 bg-blue-500/20 hover:bg-blue-500/30 text-blue-400 hover:text-blue-300 rounded-xl flex items-center justify-center transition-all duration-200 border border-blue-500/30"
                                                            title="View Details"
                                                        >
                                                            <Eye className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDownloadDocument(doc.doc_id, doc.name);
                                                            }}
                                                            className="w-10 h-10 bg-green-500/20 hover:bg-green-500/30 text-green-400 hover:text-green-300 rounded-xl flex items-center justify-center transition-all duration-200 border border-green-500/30"
                                                            title="Open PDF"
                                                        >
                                                            <ExternalLink className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDeleteDocument(doc.doc_id);
                                                            }}
                                                            className="w-10 h-10 bg-red-500/20 hover:bg-red-500/30 text-red-400 hover:text-red-300 rounded-xl flex items-center justify-center transition-all duration-200 border border-red-500/30"
                                                            title="Delete"
                                                        >
                                                            <Trash2 className="w-4 h-4" />
                                                        </button>
                                                    </div>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                )}

                {/* Users Tab */}
                {activeTab === "users" && (
                    <div className="space-y-8 animate-in fade-in-0 duration-500">
                        <div className="text-center">
                            <h2 className="text-4xl font-bold bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#6699ee] bg-clip-text text-transparent mb-4">
                                User Management
                            </h2>
                            <p className="text-gray-400 text-lg">Manage system users and permissions</p>
                        </div>

                        <div className="bg-black/20 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                            <h3 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
                                <User className="w-6 h-6 text-blue-400" />
                                Current User
                            </h3>
                            <div className="flex items-center gap-6 p-6 bg-gradient-to-r from-blue-500/10 to-purple-500/10 rounded-2xl border border-blue-500/20">
                                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl flex items-center justify-center shadow-lg">
                                    <Users className="w-8 h-8 text-white" />
                                </div>
                                <div className="flex-1">
                                    <h4 className="text-xl font-bold text-white mb-1">{user.username}</h4>
                                    <p className="text-gray-300 mb-2">{user.email}</p>
                                    <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold bg-green-500/20 text-green-300 border border-green-500/30">
                                        Administrator
                                    </span>
                                </div>
                                <div className="text-right">
                                    <p className="text-gray-400 text-sm">Access Level</p>
                                    <p className="text-white font-semibold">Full Control</p>
                                </div>
                            </div>
                        </div>

                        <div className="bg-black/20 backdrop-blur-xl p-8 rounded-3xl border border-white/10">
                            <h3 className="text-xl font-bold text-white mb-6 flex items-center gap-3">
                                <Server className="w-6 h-6 text-purple-400" />
                                Admin Configuration
                            </h3>
                            <div className="space-y-4">
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                                    <h4 className="text-white font-semibold mb-2">Authentication Method</h4>
                                    <p className="text-gray-400">CAS (Central Authentication Service)</p>
                                </div>
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                                    <h4 className="text-white font-semibold mb-2">Admin Users</h4>
                                    <p className="text-gray-400">Configured through environment variables. Contact your system administrator to modify admin access.</p>
                                </div>
                                <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                                    <h4 className="text-white font-semibold mb-2">Session Management</h4>
                                    <p className="text-gray-400">Automatic session validation with secure cookie-based authentication.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Document Detail Modal */}
                {showDocumentDetail && selectedDocument && (
                    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 p-4">
                        <div className="bg-black/40 backdrop-blur-2xl rounded-3xl border border-white/20 max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
                            <div className="sticky top-0 bg-black/60 backdrop-blur-xl p-6 border-b border-white/10 rounded-t-3xl">
                                <div className="flex justify-between items-start">
                                    <div>
                                        <h3 className="text-2xl font-bold text-white mb-2">Document Details</h3>
                                        <p className="text-gray-400">Complete information and metadata</p>
                                    </div>
                                    <button
                                        onClick={() => setShowDocumentDetail(false)}
                                        className="w-12 h-12 bg-red-500/20 hover:bg-red-500/30 text-red-400 hover:text-red-300 rounded-2xl flex items-center justify-center transition-colors"
                                    >
                                        <X className="w-6 h-6" />
                                    </button>
                                </div>
                            </div>
                            
                            <div className="p-8 space-y-6">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <FileText className="w-5 h-5" />
                                            <span className="font-medium">Document Name</span>
                                        </div>
                                        <p className="text-white font-semibold text-lg">{selectedDocument.name}</p>
                                    </div>
                                    
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <Tag className="w-5 h-5" />
                                            <span className="font-medium">Category</span>
                                        </div>
                                        <span className={`inline-flex items-center px-4 py-2 rounded-xl text-sm font-semibold border ${
                                            selectedDocument.category === 'academics' ? 'bg-blue-500/20 text-blue-300 border-blue-500/30' :
                                            selectedDocument.category === 'student' ? 'bg-green-500/20 text-green-300 border-green-500/30' :
                                            selectedDocument.category === 'faculty' ? 'bg-purple-500/20 text-purple-300 border-purple-500/30' :
                                            selectedDocument.category === 'hostel' ? 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30' :
                                            selectedDocument.category === 'mess' ? 'bg-red-500/20 text-red-300 border-red-500/30' :
                                            'bg-gray-500/20 text-gray-300 border-gray-500/30'
                                        }`}>
                                            {selectedDocument.category}
                                        </span>
                                    </div>
                                    
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <Hash className="w-5 h-5" />
                                            <span className="font-medium">Chunks</span>
                                        </div>
                                        <p className="text-white font-semibold text-lg">{selectedDocument.chunk_count || 0}</p>
                                    </div>
                                    
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <Database className="w-5 h-5" />
                                            <span className="font-medium">Embeddings</span>
                                        </div>
                                        <p className="text-white font-semibold text-lg">{selectedDocument.embedding_count || selectedDocument.chunk_count || 0}</p>
                                    </div>
                                    
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <Calendar className="w-5 h-5" />
                                            <span className="font-medium">Created</span>
                                        </div>
                                        <p className="text-white font-semibold">{new Date(selectedDocument.created_at).toLocaleString()}</p>
                                    </div>
                                    
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <User className="w-5 h-5" />
                                            <span className="font-medium">Author</span>
                                        </div>
                                        <p className="text-white font-semibold">{selectedDocument.author || 'System'}</p>
                                    </div>
                                </div>
                                
                                {selectedDocument.description && (
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-3">
                                            <Info className="w-5 h-5" />
                                            <span className="font-medium">Description</span>
                                        </div>
                                        <p className="text-white leading-relaxed">{selectedDocument.description}</p>
                                    </div>
                                )}
                                
                                {selectedDocument.metadata && Object.keys(selectedDocument.metadata).length > 0 && (
                                    <div className="bg-white/5 p-6 rounded-2xl border border-white/10">
                                        <div className="flex items-center gap-3 text-gray-400 text-sm mb-4">
                                            <Folder className="w-5 h-5" />
                                            <span className="font-medium">Metadata</span>
                                        </div>
                                        <div className="space-y-3">
                                            {Object.entries(selectedDocument.metadata).map(([key, value]) => (
                                                <div key={key} className="flex justify-between items-center py-2 border-b border-white/10 last:border-0">
                                                    <span className="text-gray-400 font-medium">{key}:</span>
                                                    <span className="text-white font-semibold">{String(value)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Action Buttons */}
                                <div className="flex justify-end gap-4 pt-4 border-t border-white/10">
                                    <button
                                        onClick={() => handleDownloadDocument(selectedDocument.doc_id, selectedDocument.name)}
                                        className="flex items-center gap-2 px-6 py-3 bg-gradient-to-r from-green-500/20 to-emerald-500/20 hover:from-green-500/30 hover:to-emerald-500/30 text-green-300 hover:text-green-200 rounded-2xl transition-all duration-200 border border-green-500/30"
                                    >
                                        <ExternalLink className="w-4 h-4" />
                                        Open PDF
                                    </button>
                                    <button
                                        onClick={() => setShowDocumentDetail(false)}
                                        className="flex items-center gap-2 px-6 py-3 bg-white/10 hover:bg-white/20 text-gray-300 hover:text-white rounded-2xl transition-all duration-200 border border-white/20"
                                    >
                                        Close
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}