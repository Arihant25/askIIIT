"use client";

import { RefreshCw, FileText, Database, Activity, Server, CheckCircle, Play, Pause } from "lucide-react";

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

interface OverviewTabProps {
    systemInfo: SystemInfo | null;
    autoRefresh: boolean;
    setAutoRefresh: (value: boolean) => void;
    loadSystemInfo: () => void;
    loadDocuments: () => void;
    loadLogs: () => void;
    handleReindex: () => void;
    handleProcessUploaded: () => void;
    processing: boolean;
    setError: (error: string) => void;
}

export default function OverviewTab({
    systemInfo,
    autoRefresh,
    setAutoRefresh,
    loadSystemInfo,
    loadDocuments,
    loadLogs,
    handleReindex,
    handleProcessUploaded,
    processing,
    setError
}: OverviewTabProps) {
    return (
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
    );
}
