"use client";

import { useState, useEffect } from "react";
import {
    Users,
    FileText,
    Database,
    Settings,
    LogOut,
    Upload,
    Trash2,
    RefreshCw,
    AlertCircle,
    CheckCircle,
    Activity,
    BarChart3
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
}

interface Document {
    doc_id: string;
    name: string;
    category: string;
    description: string;
    created_at: string;
    author?: string;
}

export default function AdminDashboard({ user }: AdminDashboardProps) {
    const [activeTab, setActiveTab] = useState("overview");
    const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
    const [documents, setDocuments] = useState<Document[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const [uploadFile, setUploadFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const [showUploadPanel, setShowUploadPanel] = useState(false);

    useEffect(() => {
        loadSystemInfo();
        loadDocuments();
    }, []);

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

    const handleUpload = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!uploadFile) return;

        setUploading(true);
        try {
            const formData = new FormData();
            formData.append("file", uploadFile);

            await AdminAPIService.uploadDocument(formData);

            setUploadFile(null);
            loadDocuments();
            loadSystemInfo();
        } catch (err) {
            setError(err instanceof Error ? err.message : "Upload failed");
        } finally {
            setUploading(false);
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
        { id: "users", label: "Users", icon: Users },
        { id: "settings", label: "Settings", icon: Settings },
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
                        <h2 className="text-2xl font-bold text-white">System Overview</h2>

                        {systemInfo && (
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                                <div className="bg-[#232946] p-6 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <FileText className="w-8 h-8 text-[#60a5fa]" />
                                        <div>
                                            <p className="text-gray-400 text-sm">Total Documents</p>
                                            <p className="text-2xl font-bold text-white">{systemInfo.documents.total}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-6 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <Database className="w-8 h-8 text-[#22c55e]" />
                                        <div>
                                            <p className="text-gray-400 text-sm">Total Chunks</p>
                                            <p className="text-2xl font-bold text-white">{systemInfo.chunks.total}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-6 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <Activity className={`w-8 h-8 ${systemInfo.models.status === 'healthy' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`} />
                                        <div>
                                            <p className="text-gray-400 text-sm">Model Status</p>
                                            <p className="text-lg font-bold text-white capitalize">{systemInfo.models.status}</p>
                                        </div>
                                    </div>
                                </div>

                                <div className="bg-[#232946] p-6 rounded-lg">
                                    <div className="flex items-center gap-3">
                                        <CheckCircle className={`w-8 h-8 ${systemInfo.database.status === 'healthy' ? 'text-[#22c55e]' : 'text-[#ef4444]'}`} />
                                        <div>
                                            <p className="text-gray-400 text-sm">Database</p>
                                            <p className="text-lg font-bold text-white">{systemInfo.database.type}</p>
                                        </div>
                                    </div>
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
                            <div className="flex gap-4">
                                <button
                                    onClick={handleReindex}
                                    className="flex items-center gap-2 bg-[#60a5fa] hover:bg-[#3b82f6] text-white px-4 py-2 rounded-lg transition-colors"
                                >
                                    <RefreshCw className="w-4 h-4" />
                                    Reindex Documents
                                </button>
                            </div>
                        </div>
                    </div>
                )}

                {activeTab === "documents" && (
                    <div className="space-y-6">
                        <h2 className="text-2xl font-bold text-white">Document Management</h2>

                        <div className="bg-[#232946] rounded-lg overflow-hidden">
                            <div className="overflow-x-auto">
                                <table className="w-full">
                                    <thead className="bg-[#181A20]">
                                        <tr>
                                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">
                                                Name
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
                                            <tr key={doc.doc_id}>
                                                <td className="px-6 py-4 whitespace-nowrap">
                                                    <div className="text-sm font-medium text-white">{doc.name}</div>
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                                                    {new Date(doc.created_at).toLocaleDateString()}
                                                </td>
                                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                                    <button
                                                        onClick={() => handleDeleteDocument(doc.doc_id)}
                                                        className="text-red-400 hover:text-red-300 transition-colors"
                                                    >
                                                        <Trash2 className="w-4 h-4" />
                                                    </button>
                                                </td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </div>

                        <button
                            onClick={() => document.getElementById('uploadFileInput')?.click()}
                            className="mt-4 bg-[#60a5fa] hover:bg-[#3b82f6] text-white font-medium py-2 px-4 rounded-lg transition-colors"
                        >
                            Upload New Document
                        </button>

                        <input
                            id="uploadFileInput"
                            type="file"
                            accept=".pdf,.txt,.doc,.docx"
                            onChange={(e) => setUploadFile(e.target.files?.[0] || null)}
                            className="hidden"
                        />
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

                {activeTab === "settings" && (
                    <div className="space-y-6">
                        <h2 className="text-2xl font-bold text-white">System Settings</h2>

                        <div className="bg-[#232946] p-6 rounded-lg">
                            <h3 className="text-lg font-semibold text-white mb-4">Configuration</h3>
                            <div className="space-y-4">
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                    <div>
                                        <label className="block text-sm font-medium text-gray-300 mb-2">
                                            Allowed File Extensions
                                        </label>
                                        <input
                                            type="text"
                                            defaultValue=".pdf,.txt,.doc,.docx"
                                            className="block w-full px-3 py-2 bg-[#181A20] border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-[#60a5fa] focus:border-transparent"
                                            disabled
                                        />
                                    </div>
                                    <div>
                                        <label className="block text-sm font-medium text-gray-300 mb-2">
                                            Max Upload Size (MB)
                                        </label>
                                        <input
                                            type="number"
                                            defaultValue="100"
                                            className="block w-full px-3 py-2 bg-[#181A20] border border-gray-600 rounded-lg text-white focus:ring-2 focus:ring-[#60a5fa] focus:border-transparent"
                                            disabled
                                        />
                                    </div>
                                </div>
                                <p className="text-sm text-gray-400">
                                    Settings are currently read-only. Modify environment variables to change configuration.
                                </p>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
