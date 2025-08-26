"use client";

import { X, Upload, FileText, Play, Tag, Hash, Database, Calendar, User, Info, Folder } from "lucide-react";
import { useState } from "react";

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

interface DocumentsTabProps {
    documents: Document[];
    loading: boolean;
    uploadFiles: File[];
    setUploadFiles: (files: File[]) => void;
    uploading: boolean;
    processing: boolean;
    handleMultipleUpload: () => Promise<void>;
    handleProcessUploaded: () => Promise<void>;
    handleDownloadDocument: (docId: string, filename: string) => Promise<void>;
    handleDeleteDocument: (docId: string) => Promise<void>;
}

export default function DocumentsTab({
    documents,
    loading,
    uploadFiles,
    setUploadFiles,
    uploading,
    processing,
    handleMultipleUpload,
    handleProcessUploaded,
    handleDownloadDocument,
    handleDeleteDocument
}: DocumentsTabProps) {
    const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);
    const [showDocumentDetail, setShowDocumentDetail] = useState(false);

    return (
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
                                            onClick={() => setUploadFiles(uploadFiles.filter((_, i: number) => i !== index))}
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
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${doc.category === 'academics' ? 'bg-blue-500/20 text-blue-400' :
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
                                        <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${doc.status === 'processed' ? 'bg-green-500/20 text-green-400' :
                                                doc.status === 'processing' ? 'bg-yellow-500/20 text-yellow-400' :
                                                    doc.status === 'failed' ? 'bg-red-500/20 text-red-400' :
                                                        'bg-gray-500/20 text-gray-400'
                                            }`}>
                                            {doc.status || 'unknown'}
                                        </span>
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-400">
                                        {new Date(doc.created_at).toLocaleDateString()}
                                    </td>
                                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                        <div className="flex items-center space-x-2">
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDownloadDocument(doc.doc_id, doc.name);
                                                }}
                                                className="text-blue-400 hover:text-blue-300"
                                            >
                                                Download
                                            </button>
                                            <button
                                                onClick={(e) => {
                                                    e.stopPropagation();
                                                    handleDeleteDocument(doc.doc_id);
                                                }}
                                                className="text-red-400 hover:text-red-300"
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

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
                                    <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${selectedDocument.category === 'academics' ? 'bg-blue-500/20 text-blue-400' :
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
                                                <span className="text-white">{typeof value === 'object' ? JSON.stringify(value) : value}</span>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}
