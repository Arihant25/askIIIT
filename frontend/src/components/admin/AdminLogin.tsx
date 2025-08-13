"use client";

import { Shield, AlertCircle } from "lucide-react";

interface AdminLoginProps {
    onLogin: () => void;
    error: string;
}

export default function AdminLogin({ onLogin, error }: AdminLoginProps) {
    return (
        <div className="min-h-screen bg-[#181A20] flex items-center justify-center">
            <div className="bg-[#232946] p-8 rounded-lg shadow-xl max-w-md w-full mx-4">
                <div className="text-center mb-8">
                    <div className="inline-flex items-center justify-center w-16 h-16 bg-[#60a5fa]/20 rounded-full mb-4">
                        <Shield className="w-8 h-8 text-[#60a5fa]" />
                    </div>
                    <h1 className="text-2xl font-bold text-white mb-2">Admin Access</h1>
                    <p className="text-gray-400">
                        Sign in with your IIIT credentials to access the admin dashboard
                    </p>
                </div>

                {error && (
                    <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg flex items-center gap-2">
                        <AlertCircle className="w-4 h-4 text-red-400" />
                        <span className="text-red-400 text-sm">{error}</span>
                    </div>
                )}

                <button
                    onClick={onLogin}
                    className="w-full bg-[#60a5fa] hover:bg-[#3b82f6] text-white font-medium py-3 px-4 rounded-lg transition-colors"
                >
                    Sign in with CAS
                </button>

                <div className="mt-6 text-center">
                    <p className="text-xs text-gray-500">
                        Only authorized administrators can access this area
                    </p>
                </div>
            </div>
        </div>
    );
}
