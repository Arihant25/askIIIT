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
    Clock,
    Hash,
    Calendar,
    User,
    Folder,
    Tag,
    Download,
    ExternalLink
} from "lucide-react";
import { AdminAPIService } from "@/services/adminAPI";

interface User {
    email: string;
    username: string;
    is_admin: boolean;
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
    };
    backend: {
        status: string;
        uptime: string;
        memory_usage?: number;
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

export default function AdminDashboard({ user }: AdminDashboardProps) {
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
            }, 5000); // Refresh every 5 seconds
            return () => clearInterval(interval);
        }
    }, [autoRefresh, logFilter, logLevel]);

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
            params.append('lines', '500'); // Get more logs
            if (logLevel) params.append('level', logLevel);
            
            const response = await fetch(`/api/admin/logs?${params.toString()}`, {
                credentials: 'include',
            });
            
            if (response.ok) {
                const data = await response.json();
                let filteredLogs = data.logs || [];
                
                // Apply text filter if specified
                if (logFilter) {
                    filteredLogs = filteredLogs.filter((log: LogEntry) => 
                        log.message.toLowerCase().includes(logFilter.toLowerCase()) ||
                        log.source.toLowerCase().includes(logFilter.toLowerCase())
                    );
                }
                
                setLogs(filteredLogs);
            }
        } catch (err) {
            // Silent fail for logs
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
            setError(""); // Clear any previous errors
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
        <div className="min-h-screen bg-[#181A20]">
            {/* Header */}
            <header className="bg-[#232946] border-b border-[#232946]/40">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="flex justify-between items-center h-16">
                        <div className="flex items-center">
                            <h1 className="text-xl font-bold text-white">Admin Dashboard</h1>
                        </div>
                        <div className="flex items-center gap-4">
                            <span className="text-gray-400 text-sm">Welcome, {user.full_name || user.name || user.username}</span>
                            <button
                                onClick={() => window.location.href = "/api/auth/logout"}
                                className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
                            >
                                <LogOut className="w-4 h-4" />
                                Logout
                            </button>
                        </div>
                    </div>
                </div>
            </header>

            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
                {/* Error Alert */}
                {error && (
                    <div className="mb-6 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2">
                        <AlertCircle className="w-5 h-5 text-red-400" />
                        <span className="text-red-400">{error}</span>
                        <button
                            onClick={() => setError("")}
                            className="ml-auto text-red-400 hover:text-red-300"
                        >
                            ×
                        </button>
                    </div>
                )}

                {/* Navigation Tabs */}
                <div className="mb-8">
                    <nav className="flex space-x-8">
                        {tabs.map((tab) => (
                            <button
                                key={tab.id}
                                onClick={() => setActiveTab(tab.id)}
                                className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg transition-colors ${activeTab === tab.id
                                    ? "bg-[#60a5fa] text-white"
                                    : "text-gray-400 hover:text-white hover:bg-[#232946]"
                                    }`}
                            >
                                <tab.icon className="w-4 h-4" />
                                {tab.label}
                            </button>
                        ))}
                    </nav>
                </div>

                {/* Tab Content */}
                {activeTab === "overview" && (
                    <div className="space-y-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-2xl font-bold text-white">System Overview</h2>
                            <div className="flex items-center gap-4">
                                <label className="flex items-center gap-2 text-sm text-gray-400">
                                    <input
                                        type="checkbox"
                                        checked={autoRefresh}
                                        onChange={(e) => setAutoRefresh(e.target.checked)}
                                        className="rounded bg-[#232946] border-gray-600"
                                    />
                                    Auto-refresh
                                </label>
                                <button
                                    onClick={() => {
                                        loadSystemInfo();
                                        loadDocuments();
                                        loadLogs();
                                    }}
                                    className="flex items-center gap-2 bg-[#60a5fa] hover:bg-[#3b82f6] text-white px-3 py-1 rounded text-sm transition-colors"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Refresh
                                </button>
                            </div>
                        </div>

                        {systemInfo && (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                                <div className="bg-[#232946] p-4 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <FileText className="w-6 h-6 text-[#60a5fa]" />
                                        <div>
                                            <p className="text-gray-400 text-xs">Documents</p>
                                            <p className="text-xl font-bold text-white">{systemInfo.documents.total}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-4 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <Database className="w-6 h-6 text-[#22c55e]" />
                                        <div>
                                            <p className="text-gray-400 text-xs">Chunks</p>
                                            <p className="text-xl font-bold text-white">{systemInfo.chunks.total}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-4 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <Activity className={`w-6 h-6 ${systemInfo.models.status === 'healthy' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`} />
                                        <div>
                                            <p className="text-gray-400 text-xs">Models</p>
                                            <p className="text-sm font-bold text-white capitalize">{systemInfo.models.status}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-4 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <Server className={`w-6 h-6 ${systemInfo.backend.status === 'healthy' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`} />
                                        <div>
                                            <p className="text-gray-400 text-xs">Backend</p>
                                            <p className="text-sm font-bold text-white capitalize">{systemInfo.backend.status}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-4 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <CheckCircle className={`w-6 h-6 ${systemInfo.database.status === 'healthy' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`} />
                                        <div>
                                            <p className="text-gray-400 text-xs">Database</p>
                                            <p className="text-sm font-bold text-white">{systemInfo.database.type}</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )}

                        {/* Processing Status */}
                        {systemInfo?.processing && (
                            <div className="bg-[#232946] p-6 rounded-lg">
                                <h3 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
                                    <Server className="w-5 h-5" />
                                    Processing Status
                                </h3>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-3 h-3 rounded-full ${systemInfo.processing.is_indexing ? 'bg-green-500 animate-pulse' : 'bg-gray-500'}`}></div>
                                        <div>
                                            <p className="text-gray-400 text-sm">Status</p>
                                            <p className="text-white font-medium">
                                                {systemInfo.processing.is_indexing ? 'Processing' : 'Idle'}
                                            </p>
                                        </div>
                                    </div>
                                    {systemInfo.processing.current_document && (
                                        <div>
                                            <p className="text-gray-400 text-sm">Current Document</p>
                                            <p className="text-white font-medium truncate">{systemInfo.processing.current_document}</p>
                                        </div>
                                    )}
                                    {systemInfo.processing.queue_size !== undefined && (
                                        <div>
                                            <p className="text-gray-400 text-sm">Queue Size</p>
                                            <p className="text-white font-medium">{systemInfo.processing.queue_size}</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}

                        {/* Categories Breakdown */}
                        {systemInfo && (
                            <div className="bg-[#232946] p-6 rounded-lg">
                                <h3 className="text-lg font-semibold text-white mb-4">Documents by Category</h3>
                                <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                                    {Object.entries(systemInfo.documents.by_category).map(([category, count]) => (
                                        <div key={category} className="text-center">
                                            <p className="text-2xl font-bold text-[#60a5fa]">{count}</p>
                                            <p className="text-sm text-gray-400 capitalize">{category}</p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {/* Actions */}
                        <div className="bg-[#232946] p-6 rounded-lg">
                            <h3 className="text-lg font-semibold text-white mb-4">System Actions</h3>
                            <div className="flex flex-wrap gap-4">
                                <button
                                    onClick={handleReindex}
                                    className="flex items-center gap-2 bg-[#60a5fa] hover:bg-[#3b82f6] text-white px-4 py-2 rounded-lg transition-colors"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Reindex Documents
                                </button>
                                <button
                                    onClick={handleProcessUploaded}
                                    disabled={processing}
                                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                                >
                                    {processing ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                                    {processing ? 'Processing...' : 'Start Bulk Processing'}
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === "documents" && (
                    <div className="space-y-6">
                        <h2 className="text-2xl font-bold text-white">Document Management</h2>

                        {/* Upload Section */}
                        <div className="bg-[#232946] p-6 rounded-lg">
                            <h3 className="text-lg font-semibold text-white mb-4">Upload Documents</h3>
                            <div className="space-y-4">
                                <div className="flex items-center gap-4">
                                    <input
                                        id="uploadFileInput"
                                        type="file"
                                        accept=".pdf"
                                        multiple
                                        onChange={(e) => setUploadFiles(Array.from(e.target.files || []))}
                                        className="block text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-[#60a5fa] file:text-white hover:file:bg-[#3b82f6] file:cursor-pointer cursor-pointer"
                                    />
                                </div>
                                {uploadFiles.length > 0 && (
                                    <div className="space-y-2">
                                        <p className="text-sm text-gray-400">Selected files ({uploadFiles.length}):</p>
                                        <div className="max-h-32 overflow-y-auto space-y-1">
                                            {uploadFiles.map((file, index) => (
                                                <div key={index} className="flex items-center justify-between bg-[#181A20] p-2 rounded text-sm">
                                                    <span className="text-white truncate">{file.name}</span>
                                                    <button
                                                        onClick={() => setUploadFiles(files => files.filter((_, i) => i !== index))}
                                                        className="text-red-400 hover:text-red-300 ml-2"
                                                    >
                                                        <X className="w-4 h-4" />
                                                    </button>
                                                </div>
                                            ))}
                                        </div>
                                        <div className="flex gap-2">
                                            <button
                                                onClick={handleMultipleUpload}
                                                disabled={uploading}
                                                className="flex items-center gap-2 bg-[#60a5fa] hover:bg-[#3b82f6] disabled:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                                            >
                                                <Upload className="w-4 h-4" />
                                                {uploading ? 'Uploading...' : 'Upload Files'}
                                            </button>
                                            <button
                                                onClick={handleProcessUploaded}
                                                disabled={processing || uploadFiles.length === 0}
                                                className="flex items-center gap-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg transition-colors"
                                            >
                                                <Play className="w-4 h-4" />
                                                {processing ? 'Processing...' : 'Process Documents'}
                                            </button>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Documents Table */}
                        <div className="bg-[#232946] rounded-lg overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead className="bg-[#181A20]">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Name
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Category
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Chunks
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Status
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Created
                                            </th>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Actions
                                            </th>
                                        </tr>
                                    </thead>
                                    <tbody className="divide-y divide-[#181A20]">
                                        {documents.map((doc) => (
                                            <tr 
                                                key={doc.doc_id} 
                                                className="hover:bg-[#181A20]/50 cursor-pointer"
                                                onClick={() => {
                                                    setSelectedDocument(doc);
                                                    setShowDocumentDetail(true);
                                                }}
                                            >
                                                <td className="px-6 py-4">
                                                    <div className="text-sm font-medium text-white truncate max-w-xs">{doc.name}</div>
                                                </td>
                                                <td className="px-6 py-4">
                                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                                        doc.category === 'academics' ? 'bg-blue-500/20 text-blue-400' :
                                                        doc.category === 'student' ? 'bg-green-500/20 text-green-400' :
                                                        doc.category === 'faculty' ? 'bg-purple-500/20 text-purple-400' :
                                                        doc.category === 'hostel' ? 'bg-yellow-500/20 text-yellow-400' :
                                                        doc.category === 'mess' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-gray-500/20 text-gray-400'
                                                    }`}>
                                                        {doc.category}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 text-sm text-gray-400">
                                                    {doc.chunk_count || 0}
                                                </td>
                                                <td className="px-6 py-4">
                                                    <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                                                        doc.status === 'processed' ? 'bg-green-500/20 text-green-400' :
                                                        doc.status === 'processing' ? 'bg-yellow-500/20 text-yellow-400' :
                                                        doc.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-gray-500/20 text-gray-400'
                                                    }`}>
                                                        {doc.status || 'processed'}
                                                    </span>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                                                    {new Date(doc.created_at).toLocaleDateString()}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                                    <div className="flex items-center gap-2">
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                setSelectedDocument(doc);
                                                                setShowDocumentDetail(true);
                                                            }}
                                                            className="text-blue-400 hover:text-blue-300 transition-colors"
                                                            title="View Details"
                                                        >
                                                            <Eye className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDownloadDocument(doc.doc_id, doc.name);
                                                            }}
                                                            className="text-green-400 hover:text-green-300 transition-colors"
                                                            title="Open PDF"
                                                        >
                                                            <ExternalLink className="w-4 h-4" />
                                                        </button>
                                                        <button
                                                            onClick={(e) => {
                                                                e.stopPropagation();
                                                                handleDeleteDocument(doc.doc_id);
                                                            }}
                                                            className="text-red-400 hover:text-red-300 transition-colors"
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

                {/* Document Detail Modal */}
                {showDocumentDetail && selectedDocument && (
                    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
                        <div className="bg-[#232946] rounded-lg p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-y-auto">
                            <div className="flex justify-between items-start mb-6">
                                <h3 className="text-xl font-bold text-white">Document Details</h3>
                                <button
                                    onClick={() => setShowDocumentDetail(false)}
                                    className="text-gray-400 hover:text-white"
                                >
                                    <X className="w-6 h-6" />
                                </button>
                            </div>
                            
                            <div className="space-y-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <FileText className="w-4 h-4" />
                                            <span>Document Name</span>
                                        </div>
                                        <p className="text-white font-medium">{selectedDocument.name}</p>
                                    </div>
                                    
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <Tag className="w-4 h-4" />
                                            <span>Category</span>
                                        </div>
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                            selectedDocument.category === 'academics' ? 'bg-blue-500/20 text-blue-400' :
                                            selectedDocument.category === 'student' ? 'bg-green-500/20 text-green-400' :
                                            selectedDocument.category === 'faculty' ? 'bg-purple-500/20 text-purple-400' :
                                            selectedDocument.category === 'hostel' ? 'bg-yellow-500/20 text-yellow-400' :
                                            selectedDocument.category === 'mess' ? 'bg-red-500/20 text-red-400' :
                                            'bg-gray-500/20 text-gray-400'
                                        }`}>
                                            {selectedDocument.category}
                                        </span>
                                    </div>
                                    
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <Hash className="w-4 h-4" />
                                            <span>Chunks</span>
                                        </div>
                                        <p className="text-white font-medium">{selectedDocument.chunk_count || 0}</p>
                                    </div>
                                    
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <Database className="w-4 h-4" />
                                            <span>Embeddings</span>
                                        </div>
                                        <p className="text-white font-medium">{selectedDocument.embedding_count || selectedDocument.chunk_count || 0}</p>
                                    </div>
                                    
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <Calendar className="w-4 h-4" />
                                            <span>Created</span>
                                        </div>
                                        <p className="text-white font-medium">{new Date(selectedDocument.created_at).toLocaleString()}</p>
                                    </div>
                                    
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <User className="w-4 h-4" />
                                            <span>Author</span>
                                        </div>
                                        <p className="text-white font-medium">{selectedDocument.author || 'System'}</p>
                                    </div>
                                </div>
                                
                                {selectedDocument.description && (
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <Info className="w-4 h-4" />
                                            <span>Description</span>
                                        </div>
                                        <p className="text-white">{selectedDocument.description}</p>
                                    </div>
                                )}
                                
                                {selectedDocument.metadata && Object.keys(selectedDocument.metadata).length > 0 && (
                                    <div className="bg-[#181A20] p-4 rounded-lg">
                                        <div className="flex items-center gap-2 text-gray-400 text-sm mb-2">
                                            <Folder className="w-4 h-4" />
                                            <span>Metadata</span>
                                        </div>
                                        <div className="space-y-2 text-sm">
                                            {Object.entries(selectedDocument.metadata).map(([key, value]) => (
                                                <div key={key} className="flex justify-between">
                                                    <span className="text-gray-400">{key}:</span>
                                                    <span className="text-white">{String(value)}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    </div>
                )}

                {/* Logs Tab */}
                {activeTab === "logs" && (
                    <div className="space-y-6">
                        <div className="flex justify-between items-center">
                            <h2 className="text-2xl font-bold text-white">System Logs</h2>
                            <div className="text-sm text-gray-400">
                                {logs.length} entries
                            </div>
                        </div>
                        
                        {/* Log Filters */}
                        <div className="bg-[#232946] p-4 rounded-lg">
                            <div className="flex flex-wrap items-center gap-4">
                                <div className="flex items-center gap-2">
                                    <label className="text-sm text-gray-400">Filter:</label>
                                    <input
                                        type="text"
                                        placeholder="Search logs..."
                                        value={logFilter}
                                        onChange={(e) => setLogFilter(e.target.value)}
                                        className="px-3 py-1 bg-[#181A20] border border-gray-600 rounded text-white text-sm focus:ring-2 focus:ring-[#60a5fa] focus:border-transparent"
                                    />
                                </div>
                                <div className="flex items-center gap-2">
                                    <label className="text-sm text-gray-400">Level:</label>
                                    <select
                                        value={logLevel}
                                        onChange={(e) => setLogLevel(e.target.value)}
                                        className="px-3 py-1 bg-[#181A20] border border-gray-600 rounded text-white text-sm focus:ring-2 focus:ring-[#60a5fa] focus:border-transparent"
                                    >
                                        <option value="">All Levels</option>
                                        <option value="DEBUG">DEBUG</option>
                                        <option value="INFO">INFO</option>
                                        <option value="WARNING">WARNING</option>
                                        <option value="ERROR">ERROR</option>
                                        <option value="CRITICAL">CRITICAL</option>
                                    </select>
                                </div>
                                <button
                                    onClick={() => {
                                        setLogFilter("");
                                        setLogLevel("");
                                    }}
                                    className="px-3 py-1 bg-gray-600 hover:bg-gray-700 text-white text-sm rounded transition-colors"
                                >
                                    Clear Filters
                                </button>
                            </div>
                        </div>
                        
                        {/* Logs Display */}
                        <div className="bg-[#232946] rounded-lg overflow-hidden">
                            <div className="h-[500px] overflow-y-auto p-4">
                                <div className="space-y-1 font-mono text-sm">
                                    {logs.length === 0 ? (
                                        <div className="flex items-center justify-center h-full">
                                            <p className="text-gray-400">No logs available {logFilter || logLevel ? 'with current filters' : ''}</p>
                                        </div>
                                    ) : (
                                        logs.map((log, index) => (
                                            <div key={index} className="flex gap-3 py-2 px-2 hover:bg-[#181A20]/30 rounded">
                                                <span className="text-gray-400 text-xs whitespace-nowrap min-w-[80px]">
                                                    {new Date(log.timestamp).toLocaleTimeString()}
                                                </span>
                                                <span className={`text-xs font-bold whitespace-nowrap min-w-[60px] ${
                                                    log.level === 'ERROR' || log.level === 'CRITICAL' ? 'text-red-400' :
                                                    log.level === 'WARN' || log.level === 'WARNING' ? 'text-yellow-400' :
                                                    log.level === 'INFO' ? 'text-blue-400' :
                                                    log.level === 'DEBUG' ? 'text-gray-400' :
                                                    'text-gray-400'
                                                }`}>
                                                    {log.level}
                                                </span>
                                                <span className="text-purple-400 text-xs whitespace-nowrap min-w-[100px]">
                                                    [{log.source || 'system'}]
                                                </span>
                                                <span className="text-white text-sm flex-1 break-all">
                                                    {log.message}
                                                </span>
                                            </div>
                                        ))
                                    )}
                                </div>
                            </div>
                        </div>
                        
                        <div className="flex justify-between items-center">
                            <div className="text-sm text-gray-400">
                                Auto-refresh: {autoRefresh ? 'On' : 'Off'} | Last updated: {new Date().toLocaleTimeString()}
                            </div>
                            <button
                                onClick={loadLogs}
                                className="flex items-center gap-2 bg-[#60a5fa] hover:bg-[#3b82f6] text-white px-4 py-2 rounded-lg transition-colors"
                            >
                                <RefreshCw className="w-4 h-4" />
                                Refresh Logs
                            </button>
                        </div>
                    </div>
                )}

                {activeTab === "users" && (
                    <div className="space-y-6">
                        <h2 className="text-2xl font-bold text-white">User Management</h2>

                        <div className="bg-[#232946] p-6 rounded-lg">
                            <h3 className="text-lg font-semibold text-white mb-4">Current User</h3>
                            <div className="flex items-center gap-3 p-4 bg-[#181A20] rounded-lg">
                                <div className="w-10 h-10 bg-[#60a5fa] rounded-full flex items-center justify-center">
                                    <Users className="w-5 h-5 text-white" />
                                </div>
                                <div>
                                    <p className="text-white font-medium">{user.username}</p>
                                    <p className="text-gray-400 text-sm">{user.email}</p>
                                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-500/20 text-green-400 mt-1">
                                        Administrator
                                    </span>
                                </div>
                            </div>
                        </div>

                        <div className="bg-[#232946] p-6 rounded-lg">
                            <h3 className="text-lg font-semibold text-white mb-4">Admin Users</h3>
                            <p className="text-gray-400 text-sm">
                                Admin users are configured through environment variables. Contact your system administrator to modify admin access.
                            </p>
                        </div>
                    </div>
                )}

            </div>
        </div>
    );
}
