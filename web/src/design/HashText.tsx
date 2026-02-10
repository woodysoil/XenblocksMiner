import { useState } from 'react';
import { colors } from './tokens';

interface HashTextProps {
  text: string;
  chars?: number;
  mono?: boolean;
  copyable?: boolean;
}

export default function HashText({ text, chars = 8, mono = true, copyable = false }: HashTextProps) {
  const [copied, setCopied] = useState(false);

  const truncated = text.length > chars + 4
    ? `${text.slice(0, chars)}...${text.slice(-4)}`
    : text;

  const handleCopy = async () => {
    await navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <span
      className={`inline-flex items-center gap-1 group ${mono ? 'font-mono' : ''} text-[${colors.text.secondary}] hover:text-[${colors.text.primary}] transition-colors`}
    >
      <span title={text}>{truncated}</span>
      {copyable && (
        <button
          onClick={handleCopy}
          className="opacity-0 group-hover:opacity-100 transition-opacity text-current"
          aria-label="Copy"
        >
          {copied ? (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
              <path d="M12.4 4.7a.75.75 0 010 1.06l-5 5a.75.75 0 01-1.06 0l-2.5-2.5a.75.75 0 111.06-1.06L6.9 9.2l4.44-4.5a.75.75 0 011.06 0z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="w-3.5 h-3.5">
              <path d="M10.5 1h-7A1.5 1.5 0 002 2.5v9a.5.5 0 001 0v-9a.5.5 0 01.5-.5h7a.5.5 0 000-1zM12 3H5.5A1.5 1.5 0 004 4.5v9A1.5 1.5 0 005.5 15H12a1.5 1.5 0 001.5-1.5v-9A1.5 1.5 0 0012 3zm0 11H5.5a.5.5 0 01-.5-.5v-9a.5.5 0 01.5-.5H12a.5.5 0 01.5.5v9a.5.5 0 01-.5.5z" />
            </svg>
          )}
        </button>
      )}
    </span>
  );
}
