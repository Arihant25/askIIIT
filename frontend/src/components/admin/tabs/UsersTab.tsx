"use client";

import { Users, LogOut } from "lucide-react";

interface User {
    email: string;
    username: string;
    is_admin: boolean;
}

interface UsersTabProps {
    user: User;
}

export default function UsersTab({
    user
}: UsersTabProps) {
    return (
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
    );
}
