import React from "react";
import { Search, BookOpen, MessageCircle } from "lucide-react";

interface LandingContentProps {
  chatStarted: boolean;
  isVisible: boolean;
  activeFeature: number;
  setActiveFeature: (index: number) => void;
}

const LandingContent: React.FC<LandingContentProps> = ({
  chatStarted,
  isVisible,
}) => {

  const stats = [
    { number: "300+", label: "Documents Indexed" },
    { number: "5", label: "Categories Covered" },
    { number: "24/7", label: "Always Available" },
  ];

  return (
    <div
      className={`absolute inset-0 z-10 transition-all duration-1000 pt-16 overflow-y-auto ${
        chatStarted
          ? "opacity-0 translate-y-[-100vh] pointer-events-none"
          : "opacity-100 translate-y-0"
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 min-h-[calc(100vh-12rem)] flex flex-col justify-center">
        <div
          className={`text-center transform transition-all duration-1000 ${
            isVisible ? "translate-y-0 opacity-100" : "translate-y-10 opacity-0"
          }`}
        >
          <h1 className="text-3xl sm:text-4xl md:text-6xl lg:text-7xl font-bold mb-4 sm:mb-6 leading-tight">
            <span className="bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#6699ee] bg-clip-text text-transparent">
              Your AI Assistant
            </span>
            <br />
            <span className="bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] bg-clip-text text-transparent">
              for IIIT
            </span>
          </h1>
          
            <p>
              Jagruti is a LLM at heart, and has a tendency to make mistakes. We recommend verifying results with provided references.
            </p>

        </div>
      </div>
    </div>
  );
};

export default LandingContent;
