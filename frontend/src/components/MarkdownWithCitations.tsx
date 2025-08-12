"use client";

import React, { useEffect } from 'react';
import MarkdownRenderer from './MarkdownRenderer';

interface MarkdownWithCitationsProps {
  content: string;
  className?: string;
}

const MarkdownWithCitations: React.FC<MarkdownWithCitationsProps> = ({ content, className = '' }) => {
  // Convert citation markers [1] to clickable superscript elements
  const processContent = (text: string) => {
    return text.replace(/\[(\d+)\]/g, (match, num) => {
      return `<sup class="citation-mark citation-clickable" data-citation="${num}" onclick="window.highlightReference && window.highlightReference(${num})">${num}</sup>`;
    });
  };

  const processedContent = processContent(content);

  // If the content contains citations, render as HTML + markdown hybrid
  if (processedContent !== content) {
    return (
      <div className={`markdown-content ${className}`}>
        <div
          dangerouslySetInnerHTML={{ __html: processedContent }}
          style={{ 
            lineHeight: '1.6',
            whiteSpace: 'pre-wrap'
          }}
        />
      </div>
    );
  }

  // Otherwise, render as pure markdown
  return <MarkdownRenderer content={content} className={className} />;
};

export default MarkdownWithCitations;