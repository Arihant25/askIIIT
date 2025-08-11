import React, { useState, useEffect } from 'react';
import { ArrowRight, MessageCircle, Search, BookOpen, Clock, Sparkles } from 'lucide-react';
import '../index.css';

const ModernLandingPage = () => {
    const [isVisible, setIsVisible] = useState(false);
    const [activeFeature, setActiveFeature] = useState(0);

    useEffect(() => {
        setIsVisible(true);

        const interval = setInterval(() => {
            setActiveFeature(prev => (prev + 1) % 3);
        }, 3000);

        return () => clearInterval(interval);
    }, []);

    const features = [
        {
            icon: <Search className="w-6 h-6" />,
            title: "Instant Search",
            description: "Find answers across all IIIT documents in seconds"
        },
        {
            icon: <BookOpen className="w-6 h-6" />,
            title: "Official Sources",
            description: "Get reliable info from verified college documents"
        },
        {
            icon: <MessageCircle className="w-6 h-6" />,
            title: "AI-Powered Chat",
            description: "Ask questions naturally and get precise answers"
        }
    ];

    const stats = [
        { number: "10K+", label: "Documents Indexed" },
        { number: "50+", label: "Categories Covered" },
        { number: "24/7", label: "Always Available" }
    ];

    const categories = [
        { name: "Academics", icon: "📚", color: "bg-blue-500" },
        { name: "Campus Life", icon: "🏫", color: "bg-green-500" },
        { name: "Exams", icon: "📝", color: "bg-purple-500" },
        { name: "Placements", icon: "💼", color: "bg-orange-500" },
        { name: "Research", icon: "🔬", color: "bg-red-500" },
        { name: "Policies", icon: "📋", color: "bg-indigo-500" }
    ];

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 overflow-hidden">
            {/* Animated background elements */}
            <div className="absolute inset-0 overflow-hidden">
                <div className="absolute -top-40 -right-40 w-80 h-80 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
                <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-blue-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse delay-1000"></div>
                <div className="absolute top-40 left-1/2 w-60 h-60 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-10 animate-pulse delay-2000"></div>
            </div>

            {/* Navigation */}
            <nav className="relative z-10 p-6">
                <div className="max-w-7xl mx-auto flex justify-between items-center">
                    <div className="flex items-center space-x-2">
                        <div className="w-8 h-8 bg-gradient-to-r from-purple-400 to-pink-400 rounded-lg flex items-center justify-center">
                            <Sparkles className="w-5 h-5 text-white" />
                        </div>
                        <span className="text-2xl font-bold bg-gradient-to-r from-white to-purple-300 bg-clip-text text-transparent">
                            askIIIT
                        </span>
                    </div>
                    <button className="px-4 py-2 text-purple-300 hover:text-white transition-colors border border-purple-400 rounded-full hover:bg-purple-500/20">
                        Login
                    </button>
                </div>
            </nav>

            {/* Hero Section */}
            <div className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32">
                <div className={`text-center transform transition-all duration-1000 ${isVisible ? 'translate-y-0 opacity-100' : 'translate-y-10 opacity-0'}`}>
                    <div className="inline-flex items-center px-4 py-2 bg-purple-500/20 border border-purple-400/30 rounded-full text-purple-300 text-sm mb-8 backdrop-blur-sm">
                        <Clock className="w-4 h-4 mr-2" />
                        Available 24/7 • No Registration Required
                    </div>

                    <h1 className="text-6xl md:text-7xl font-bold mb-6 leading-tight">
                        <span className="bg-gradient-to-r from-white via-purple-200 to-pink-200 bg-clip-text text-transparent">
                            Your AI Assistant
                        </span>
                        <br />
                        <span className="bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
                            for IIIT
                        </span>
                    </h1>

                    <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto leading-relaxed">
                        Skip the endless document searches. Get instant, accurate answers about academics, policies, campus life, and more—powered by AI and official IIIT sources.
                    </p>

                    {/* CTA Buttons */}
                    <div className="flex flex-col sm:flex-row gap-4 justify-center items-center mb-16">
                        <button className="group px-8 py-4 bg-gradient-to-r from-purple-600 to-pink-600 rounded-full text-white font-semibold text-lg transition-all hover:scale-105 hover:shadow-2xl hover:shadow-purple-500/25 flex items-center">
                            Start Chatting Now
                            <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                        </button>
                        <button className="px-8 py-4 border-2 border-white/20 rounded-full text-white font-semibold text-lg backdrop-blur-sm hover:bg-white/10 transition-all flex items-center">
                            <MessageCircle className="w-5 h-5 mr-2" />
                            See How It Works
                        </button>
                    </div>

                    {/* Stats */}
                    <div className="grid grid-cols-3 gap-8 max-w-md mx-auto">
                        {stats.map((stat, index) => (
                            <div key={index} className="text-center">
                                <div className="text-2xl font-bold text-white mb-1">{stat.number}</div>
                                <div className="text-sm text-gray-400">{stat.label}</div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Features Section */}
            <div className="relative z-10 max-w-7xl mx-auto px-6 pb-20">
                <div className="grid md:grid-cols-3 gap-8">
                    {features.map((feature, index) => (
                        <div
                            key={index}
                            className={`group p-8 rounded-2xl backdrop-blur-sm border transition-all duration-500 cursor-pointer ${activeFeature === index
                                    ? 'bg-white/10 border-purple-400/50 scale-105'
                                    : 'bg-white/5 border-white/10 hover:bg-white/8'
                                }`}
                            onMouseEnter={() => setActiveFeature(index)}
                        >
                            <div className={`w-12 h-12 rounded-xl mb-4 flex items-center justify-center transition-colors ${activeFeature === index ? 'bg-purple-500' : 'bg-white/10'
                                }`}>
                                {React.cloneElement(feature.icon, {
                                    className: `w-6 h-6 ${activeFeature === index ? 'text-white' : 'text-purple-300'}`
                                })}
                            </div>
                            <h3 className="text-xl font-semibold text-white mb-3">{feature.title}</h3>
                            <p className="text-gray-300 leading-relaxed">{feature.description}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* Categories Preview */}
            <div className="relative z-10 max-w-7xl mx-auto px-6 pb-20">
                <div className="text-center mb-12">
                    <h2 className="text-3xl font-bold text-white mb-4">Ask About Anything</h2>
                    <p className="text-gray-300">Get answers across all major IIIT categories</p>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                    {categories.map((category, index) => (
                        <div
                            key={index}
                            className="group p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-all hover:scale-105 cursor-pointer"
                        >
                            <div className="text-center">
                                <div className="text-2xl mb-2">{category.icon}</div>
                                <div className="text-sm font-medium text-white">{category.name}</div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>

            {/* Bottom CTA */}
            <div className="relative z-10 max-w-4xl mx-auto px-6 pb-20 text-center">
                <div className="p-8 rounded-2xl bg-gradient-to-r from-purple-500/20 to-pink-500/20 border border-purple-400/30 backdrop-blur-sm">
                    <h2 className="text-3xl font-bold text-white mb-4">Ready to Get Started?</h2>
                    <p className="text-gray-300 mb-6">Join thousands of IIIT students getting instant answers</p>
                    <button className="group px-8 py-4 bg-white text-purple-900 rounded-full font-semibold text-lg transition-all hover:scale-105 hover:shadow-xl flex items-center mx-auto">
                        Launch askIIIT Chat
                        <ArrowRight className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
                    </button>
                </div>
            </div>
        </div>
    );
};

export default ModernLandingPage;