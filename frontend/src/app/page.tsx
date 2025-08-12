"use client";

import { useState, useEffect } from "react";
import LandingContent from "@/components/LandingContent";
import Chat from "@/components/Chat";

export default function Home() {
  const [chatStarted, setChatStarted] = useState(false);
  const [isVisible, setIsVisible] = useState(false);
  const [activeFeature, setActiveFeature] = useState(0);

  useEffect(() => {
    setIsVisible(true);

    const interval = setInterval(() => {
      setActiveFeature((prev) => (prev + 1) % 3);
    }, 3000);

    return () => {
      clearInterval(interval);
    };
  }, []);

  return (
    <div className="min-h-screen w-screen bg-[#181A20] flex flex-col relative overflow-hidden">
      {/* Landing Content */}
      <LandingContent
        chatStarted={chatStarted}
        isVisible={isVisible}
        activeFeature={activeFeature}
        setActiveFeature={setActiveFeature}
      />

      {/* Main Content Area - Chat */}
      <main className="flex-1 flex flex-col pt-20">
        <Chat chatStarted={chatStarted} setChatStarted={setChatStarted} />
      </main>
    </div>
  );
}
