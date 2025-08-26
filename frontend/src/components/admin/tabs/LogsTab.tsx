"use client";

import { useState } from "react";
import { RefreshCw, Terminal } from "lucide-react";

interface LogEntry {
    timestamp: string;
    level: string;
    message: string;
    source?: string;
}

interface LogsTabProps {
    logs: LogEntry[];
    autoRefresh: boolean;
    setAutoRefresh: (value: boolean) => void;
    loadLogs: () => Promise<void>;
}

export default function LogsTab({
    logs,
    autoRefresh,
    setAutoRefresh,
    loadLogs,
}: LogsTabProps) {
    const [logFilter, setLogFilter] = useState<string>("");
    const [logLevel, setLogLevel] = useState<string>("");

    return (
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
                            logs.filter(log => {
                                if (logFilter && !log.message.toLowerCase().includes(logFilter.toLowerCase()) &&
                                    (!log.source || !log.source.toLowerCase().includes(logFilter.toLowerCase()))) {
                                    return false;
                                }
                                if (logLevel && log.level !== logLevel) {
                                    return false;
                                }
                                return true;
                            }).map((log, index) => (
                                <div key={index} className="flex gap-3 py-2 px-2 hover:bg-[#181A20]/30 rounded">
                                    <span className="text-gray-400 text-xs whitespace-nowrap min-w-[80px]">
                                        {new Date(log.timestamp).toLocaleTimeString()}
                                    </span>
                                    <span className={`text-xs font-bold whitespace-nowrap min-w-[60px] ${log.level === 'ERROR' || log.level === 'CRITICAL' ? 'text-red-400' :
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
                    onClick={() => loadLogs()}
                    className="flex items-center gap-2 bg-[#60a5fa] hover:bg-[#3b82f6] text-white px-4 py-2 rounded-lg transition-colors"
                >
                    <RefreshCw className="w-4 h-4" />
                    Refresh Logs
                </button>
            </div>
        </div>
    );
}
