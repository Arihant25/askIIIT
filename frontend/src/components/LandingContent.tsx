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
  activeFeature,
  setActiveFeature,
}) => {
  const features = [
    {
      icon: <Search className="w-6 h-6" />,
      title: "Instant Search",
      description: "Find answers across all IIIT documents in seconds",
    },
    {
      icon: <BookOpen className="w-6 h-6" />,
      title: "Official Sources",
      description: "Get reliable info from verified college documents",
    },
    {
      icon: <MessageCircle className="w-6 h-6" />,
      title: "AI-Powered Chat",
      description: "Ask questions naturally and get precise answers",
    },
  ];

  const stats = [
    { number: "10K+", label: "Documents Indexed" },
    { number: "50+", label: "Categories Covered" },
    { number: "24/7", label: "Always Available" },
  ];

  return (
    <div
      className={`absolute inset-0 z-10 transition-all duration-1000 pt-16 pb-32 overflow-y-auto ${
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
            <span className="bg-gradient-to-r from-[#93c5fd] via-[#60a5fa] to-[#232946] bg-clip-text text-transparent">
              Your AI Assistant
            </span>
            <br />
            <span className="bg-gradient-to-r from-[#60a5fa] to-[#93c5fd] bg-clip-text text-transparent">
              for IIIT
            </span>
          </h1>

          <p className="text-base sm:text-lg md:text-xl text-[#93c5fd] mb-8 sm:mb-12 max-w-xl sm:max-w-2xl mx-auto leading-relaxed px-2">
            Skip the endless document searches. Get instant, accurate answers
            about academics, policies, campus life, and more—powered by AI and
            official IIIT sources.
          </p>

          <div className="grid grid-cols-3 gap-2 sm:gap-4 md:gap-8 max-w-xs sm:max-w-md mx-auto mb-8 sm:mb-12">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <div className="text-lg sm:text-xl md:text-2xl font-bold text-[#60a5fa] mb-1">
                  {stat.number}
                </div>
                <div className="text-xs md:text-sm text-[#93c5fd]">
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 sm:gap-6 px-2 mb-8 sm:mb-12">
          {features.map((feature, index) => (
            <div
              key={index}
              className={`group p-4 sm:p-6 md:p-8 rounded-2xl backdrop-blur-sm border transition-all duration-500 cursor-pointer ${
                activeFeature === index
                  ? "bg-[#93c5fd]/10 border-[#60a5fa]/50 scale-[1.02] sm:scale-105"
                  : "bg-[#232946]/10 border-[#93c5fd]/10 hover:bg-[#232946]/20"
              }`}
              onMouseEnter={() => setActiveFeature(index)}
              onClick={() => setActiveFeature(index)}
            >
              <div
                className={`w-10 h-10 sm:w-12 sm:h-12 rounded-xl mb-3 sm:mb-4 flex items-center justify-center transition-colors ${
                  activeFeature === index ? "bg-[#60a5fa]" : "bg-[#232946]/10"
                }`}
              >
                {feature.icon}
              </div>
              <h3 className="text-base sm:text-lg md:text-xl font-semibold text-[#93c5fd] mb-2 sm:mb-3">
                {feature.title}
              </h3>
              <p className="text-xs sm:text-sm md:text-base text-[#93c5fd] leading-relaxed">
                {feature.description}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default LandingContent;
