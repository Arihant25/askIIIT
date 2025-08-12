"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Image from "next/image";
import { Sparkles } from "lucide-react";

const Navbar = () => {
  const [backendStatus, setBackendStatus] = useState<
    "checking" | "connected" | "disconnected"
  >("checking");

  useEffect(() => {
    const checkBackendStatus = async () => {
      try {
        const response = await fetch("/api/health", {
          method: "GET",
          headers: {
            "Content-Type": "application/json",
          },
        });

        if (response.ok) {
          setBackendStatus("connected");
        } else {
          setBackendStatus("disconnected");
        }
      } catch (error) {
        console.error("Backend health check failed:", error);
        setBackendStatus("disconnected");
      }
    };

    checkBackendStatus();

    // Check status every 30 seconds
    const interval = setInterval(checkBackendStatus, 30000);

    return () => clearInterval(interval);
  }, []);

  return (
    <header className="fixed top-0 left-0 w-full bg-[#181A20]/85 backdrop-blur-md border-b border-[#232946]/40 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo and Site Name */}
          <div className="flex items-center">
            <div className="flex-shrink-0 flex items-center">
              <div className="w-8 h-8 md:w-9 md:h-9 flex items-center justify-center bg-gradient-to-r from-[#60a5fa]/20 to-[#93c5fd]/20 rounded-lg p-1.5 mr-3">
                <Sparkles className="w-full h-full text-[#60a5fa]" />
              </div>
              <Link
                href="/"
                className="text-xl md:text-2xl font-bold bg-gradient-to-r from-[#93c5fd] to-[#60a5fa] bg-clip-text text-transparent mr-4"
              >
                Jagruti
              </Link>

              {/* Status indicator moved here */}
              <div
                className={`hidden xs:flex items-center gap-2 text-xs ${
                  backendStatus === "connected"
                    ? "text-green-400"
                    : backendStatus === "disconnected"
                    ? "text-red-400"
                    : "text-yellow-400"
                }`}
              >
                <div
                  className={`w-2 h-2 rounded-full ${
                    backendStatus === "connected"
                      ? "bg-green-400"
                      : backendStatus === "disconnected"
                      ? "bg-red-400"
                      : "bg-yellow-400"
                  }`}
                ></div>
                {backendStatus === "connected"
                  ? "Connected"
                  : backendStatus === "disconnected"
                  ? "Backend Offline"
                  : "Checking..."}
              </div>
            </div>
          </div>

          {/* College Logo */}
          <div className="flex items-center">
            <div className="w-16 sm:w-22 h-12 sm:h-16 relative flex items-center justify-center">
              <Image
                src="/iiit-logo.png"
                alt="IIIT Logo"
                layout="fill"
                objectFit="contain"
              />
            </div>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Navbar;
