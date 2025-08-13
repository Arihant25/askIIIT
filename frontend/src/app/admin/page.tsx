"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import ModernAdminDashboard from "@/components/admin/ModernAdminDashboard";
import AdminLogin from "@/components/admin/AdminLogin";
import { AdminAPIService } from "@/services/adminAPI";

interface User {
    email: string;
    name: string;
    is_admin: boolean;
}

export default function AdminPage() {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState("");
    const router = useRouter();

    useEffect(() => {
        // Check for error parameters in URL
        const urlParams = new URLSearchParams(window.location.search);
        const urlError = urlParams.get('error');
        if (urlError) {
            switch (urlError) {
                case 'no_ticket':
                    setError('Authentication failed: No ticket received');
                    break;
                case 'auth_failed':
                    setError('Authentication failed: Invalid credentials');
                    break;
                case 'validation_error':
                    setError('Authentication failed: Validation error');
                    break;
                default:
                    setError('Authentication failed');
            }
            // Clean up URL
            window.history.replaceState({}, '', '/admin');
        }

        checkAuthStatus();
    }, []);

    const checkAuthStatus = async () => {
        try {
            const userData = await AdminAPIService.getUserInfo();
            if (userData.is_admin) {
                setUser(userData);
            } else {
                setError("Admin access required");
            }
        } catch (err) {
            setError("Not authenticated");
        } finally {
            setLoading(false);
        }
    };

    const handleLogin = async () => {
        try {
            const data = await AdminAPIService.getLoginURL();
            window.location.href = data.login_url;
        } catch (err) {
            setError("Failed to initiate login");
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-[#181A20] flex items-center justify-center">
                <div className="text-white">Loading...</div>
            </div>
        );
    }

    if (!user) {
        return (
            <AdminLogin
                onLogin={handleLogin}
                error={error}
            />
        );
    }

    return <ModernAdminDashboard user={user} />;
}
