'use client';
// components/ChatWindow.tsx — scrollable message list

import { useEffect, useRef } from 'react';
import MessageBubble, { Message } from './MessageBubble';
import styles from './ChatWindow.module.css';

interface Props { messages: Message[]; }

export default function ChatWindow({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to latest message
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className={styles.empty}>
        <div className={styles.emptyIcon}>🔍</div>
        <h2 className={styles.emptyTitle}>Ask your documents anything</h2>
        <p className={styles.emptySubtitle}>
          Upload your PDFs, notes, or reports and start querying them with AI.
        </p>
        <div className={styles.suggestions}>
          {EXAMPLE_QUERIES.map((q, i) => (
            <div key={i} className={styles.suggestionChip}>{q}</div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.window} role="log" aria-live="polite" aria-label="Chat messages">
      <div className={styles.messages}>
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}

const EXAMPLE_QUERIES = [
  '📋 Summarize the key findings',
  '🔍 What does the document say about X?',
  '📊 List the main recommendations',
  '❓ What is the conclusion?',
];
