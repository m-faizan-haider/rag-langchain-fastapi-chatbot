'use client';
// components/MessageBubble.tsx

import { Source, FactVerification } from '@/lib/api';
import SourcePanel from './SourcePanel';
import styles from './MessageBubble.module.css';

export type Role = 'user' | 'assistant';

export interface Message {
  id:         string;
  role:       Role;
  content:    string;
  sources?:   Source[];
  elapsed_s?: number;
  cache_hit?: boolean;
  isStreaming?: boolean;
  verification?: FactVerification[];
}

interface Props {
  message: Message;
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user';

  return (
    <div className={`${styles.wrapper} ${isUser ? styles.userWrapper : styles.assistantWrapper}`}>
      {/* Avatar */}
      <div className={`${styles.avatar} ${isUser ? styles.userAvatar : styles.botAvatar}`}>
        {isUser ? '👤' : '🤖'}
      </div>

      {/* Bubble */}
      <div className={`${styles.bubble} ${isUser ? styles.userBubble : styles.assistantBubble}`}>
        {/* Message content */}
        <div
          className={`answer-content ${styles.content} ${message.isStreaming ? 'typing-cursor' : ''}`}
          dangerouslySetInnerHTML={{ __html: formatAnswer(message.content) }}
        />

        {/* Footer — only for assistant */}
        {!isUser && !message.isStreaming && (
          <div className={styles.footer}>
            {message.elapsed_s !== undefined && (
              <span className={styles.meta}>⚡ {message.elapsed_s.toFixed(2)}s</span>
            )}
            {message.cache_hit && (
              <span className={`${styles.badge} ${styles.cacheBadge}`}>⚡ cached</span>
            )}
            {message.sources && message.sources.length > 0 && (
              <SourcePanel sources={message.sources} />
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ── Simple markdown-ish formatter ──────────────────────────────────────────
function formatAnswer(text: string): string {
  if (!text) return '';
  return text
    // Bold
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    // Inline code
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Numbered lists
    .replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>')
    // Bullet lists
    .replace(/^[-•*]\s+(.+)$/gm, '<li>$1</li>')
    // Wrap consecutive <li>s
    .replace(/(<li>.*<\/li>\n?)+/g, (m) => `<ul>${m}</ul>`)
    // Newlines → paragraphs
    .split('\n\n')
    .map((p) => p.trim())
    .filter(Boolean)
    .map((p) => (p.startsWith('<ul>') || p.startsWith('<ol>') ? p : `<p>${p}</p>`))
    .join('');
}
