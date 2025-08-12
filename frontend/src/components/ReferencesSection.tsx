"use client";

import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Download } from 'lucide-react';

interface Source {
  filename: string;
  category: string;
}

interface ReferencesSectionProps {
  sources: Source[];
}

const ReferencesSection: React.FC<ReferencesSectionProps> = ({ sources }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  if (!sources || sources.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 pt-3 border-t border-[#232946]/20">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex items-center gap-2 text-sm opacity-75 hover:opacity-100 transition-opacity duration-200 mb-2 cursor-pointer"
      >
        <span>References ({sources.length})</span>
        {isExpanded ? (
          <ChevronUp size={16} />
        ) : (
          <ChevronDown size={16} />
        )}
      </button>
      
      {isExpanded && (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 mt-2">
          {sources.map((source, idx) => (
            <a
              key={idx}
              href={`/api/download/${encodeURIComponent(source.filename)}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-2 p-2 bg-[#232946]/10 hover:bg-[#232946]/20 rounded-lg border border-[#232946]/20 hover:border-[#60a5fa]/30 transition-all duration-200 group cursor-pointer"
              title={`Download ${source.filename}`}
            >
              <div className="flex-shrink-0">
                <Download size={14} className="text-[#93c5fd] group-hover:text-[#60a5fa] transition-colors duration-200" />
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-[#232946] group-hover:text-[#60a5fa] transition-colors duration-200 truncate">
                  {source.filename}
                </div>
                <div className="text-xs opacity-60 text-[#232946]">
                  {source.category}
                </div>
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  );
};

export default ReferencesSection;