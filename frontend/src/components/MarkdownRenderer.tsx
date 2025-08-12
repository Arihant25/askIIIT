"use client";

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';

interface MarkdownRendererProps {
  content: string;
  className?: string;
}

const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content, className = '' }) => {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={{
          // Headings
          h1: ({ children }) => (
            <h1 className="text-2xl font-bold mb-4 text-[#232946] border-b border-[#232946]/20 pb-2">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="text-xl font-bold mb-3 text-[#232946] border-b border-[#232946]/10 pb-1">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-lg font-semibold mb-2 text-[#232946]">
              {children}
            </h3>
          ),
          h4: ({ children }) => (
            <h4 className="text-base font-semibold mb-2 text-[#232946]">
              {children}
            </h4>
          ),
          h5: ({ children }) => (
            <h5 className="text-sm font-semibold mb-2 text-[#232946]">
              {children}
            </h5>
          ),
          h6: ({ children }) => (
            <h6 className="text-xs font-semibold mb-2 text-[#232946]">
              {children}
            </h6>
          ),

          // Paragraphs
          p: ({ children }) => (
            <p className="mb-3 leading-relaxed text-[#232946]">
              {children}
            </p>
          ),

          // Lists
          ul: ({ children }) => (
            <ul className="list-disc list-inside mb-3 ml-4 space-y-1 text-[#232946]">
              {children}
            </ul>
          ),
          ol: ({ children }) => (
            <ol className="list-decimal list-inside mb-3 ml-4 space-y-1 text-[#232946]">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="leading-relaxed">
              {children}
            </li>
          ),

          // Code
          code: ({ inline, children, className }) => {
            if (inline) {
              return (
                <code className="bg-[#232946]/10 text-[#232946] px-1.5 py-0.5 rounded text-sm font-mono border border-[#232946]/20">
                  {children}
                </code>
              );
            }
            return (
              <code className={`${className || ''} text-sm font-mono`}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="bg-[#232946] text-[#93c5fd] p-4 rounded-lg overflow-x-auto mb-3 border border-[#232946]/20 font-mono text-sm">
              {children}
            </pre>
          ),

          // Blockquotes
          blockquote: ({ children }) => (
            <blockquote className="border-l-4 border-[#60a5fa] pl-4 py-2 mb-3 bg-[#60a5fa]/5 italic text-[#232946]">
              {children}
            </blockquote>
          ),

          // Tables
          table: ({ children }) => (
            <div className="overflow-x-auto mb-3">
              <table className="w-full border-collapse border border-[#232946]/20 rounded-lg">
                {children}
              </table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-[#232946]/10">
              {children}
            </thead>
          ),
          tbody: ({ children }) => (
            <tbody>
              {children}
            </tbody>
          ),
          tr: ({ children }) => (
            <tr className="border-b border-[#232946]/10 hover:bg-[#232946]/5">
              {children}
            </tr>
          ),
          th: ({ children }) => (
            <th className="px-4 py-2 text-left font-semibold text-[#232946] border-r border-[#232946]/10 last:border-r-0">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="px-4 py-2 text-[#232946] border-r border-[#232946]/10 last:border-r-0">
              {children}
            </td>
          ),

          // Links
          a: ({ href, children }) => (
            <a 
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#60a5fa] hover:text-[#3b82f6] underline hover:no-underline transition-colors duration-200"
            >
              {children}
            </a>
          ),

          // Horizontal rule
          hr: () => (
            <hr className="border-0 border-t border-[#232946]/20 my-6" />
          ),

          // Strong and emphasis
          strong: ({ children }) => (
            <strong className="font-bold text-[#232946]">
              {children}
            </strong>
          ),
          em: ({ children }) => (
            <em className="italic text-[#232946]">
              {children}
            </em>
          ),

          // Images
          img: ({ src, alt }) => (
            <img 
              src={src} 
              alt={alt}
              className="max-w-full h-auto rounded-lg mb-3 border border-[#232946]/20"
            />
          ),
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default MarkdownRenderer;