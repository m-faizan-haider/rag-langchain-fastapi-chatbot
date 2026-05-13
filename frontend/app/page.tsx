'use client';
// app/page.tsx — Main RAG Chat Application

import { useState, useCallback, useEffect } from 'react';
import ChatWindow from '@/components/ChatWindow';
import InputBar from '@/components/InputBar';
import { Message } from '@/components/MessageBubble';
import { queryRAG, checkHealth } from '@/lib/api';
import styles from './page.module.css';

let msgIdCounter = 0;
const newId = () => `msg-${++msgIdCounter}-${Date.now()}`;

export default function Home() {
  const [messages, setMessages]   = useState<Message[]>([]);
  const [loading, setLoading]     = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus]       = useState<'idle' | 'ok' | 'error'>('idle');
  const [modelInfo, setModelInfo] = useState('');

  // ── Health check on mount ──────────────────────────────────────────────────
  useEffect(() => {
    checkHealth()
      .then((data) => {
        setStatus('ok');
        setModelInfo(data.embedding_model ?? '');
      })
      .catch(() => setStatus('error'));
  }, []);

  // ── Send a message ─────────────────────────────────────────────────────────
  const handleSend = useCallback(async (text: string) => {
    if (loading) return;

    // Add user message immediately
    const userMsg: Message = { id: newId(), role: 'user', content: text };
    setMessages((prev) => [...prev, userMsg]);

    // Add streaming placeholder for assistant
    const botId = newId();
    const thinkingMsg: Message = {
      id: botId, role: 'assistant', content: '…', isStreaming: true,
    };
    setMessages((prev) => [...prev, thinkingMsg]);
    setLoading(true);

    try {
      const res = await queryRAG({
        question:   text,
        session_id: sessionId,
        debug:      false,
      });

      // Update session id
      if (res.session_id && !sessionId) setSessionId(res.session_id);

      // Replace placeholder with real answer
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? {
                ...m,
                content:    res.answer,
                sources:    res.sources,
                elapsed_s:  res.elapsed_s,
                cache_hit:  res.cache_hit ?? false,
                isStreaming: false,
              }
            : m
        )
      );
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Unknown error';
      setMessages((prev) =>
        prev.map((m) =>
          m.id === botId
            ? { ...m, content: `❌ Error: ${errMsg}`, isStreaming: false }
            : m
        )
      );
    } finally {
      setLoading(false);
    }
  }, [loading, sessionId]);

  // ── Clear chat ─────────────────────────────────────────────────────────────
  const handleClear = () => {
    setMessages([]);
    setSessionId(null);
  };

  return (
    <div className={styles.layout}>
      {/* ── Sidebar ────────────────────────────────────────────────────────── */}
      <aside className={styles.sidebar}>
        <div className={styles.logoBlock}>
          <div className={styles.logoIcon}>⚡</div>
          <div>
            <h1 className={styles.logoText}>RAG Chat</h1>
            <p className={styles.logoSub}>Powered by LangChain</p>
          </div>
        </div>

        {/* Status indicator */}
        <div className={styles.statusRow}>
          <span className={`${styles.statusDot} ${styles[`statusDot_${status}`]}`} />
          <span className={styles.statusLabel}>
            {status === 'ok' ? 'Backend connected' : status === 'error' ? 'Backend offline' : 'Connecting…'}
          </span>
        </div>

        {modelInfo && (
          <div className={styles.modelCard}>
            <span className={styles.modelLabel}>Embedding model</span>
            <span className={styles.modelName}>{modelInfo.split('/').pop()}</span>
          </div>
        )}

        {/* Session info */}
        {sessionId && (
          <div className={styles.sessionCard}>
            <span className={styles.sessionLabel}>Session</span>
            <code className={styles.sessionId}>{sessionId.slice(0, 8)}…</code>
          </div>
        )}

        {/* Capabilities */}
        <div className={styles.capsList}>
          <h3 className={styles.capsTitle}>Features</h3>
          {FEATURES.map((f, i) => (
            <div key={i} className={styles.capItem}>
              <span>{f.icon}</span>
              <span>{f.label}</span>
            </div>
          ))}
        </div>

        {/* Clear button */}
        {messages.length > 0 && (
          <button className={styles.clearBtn} onClick={handleClear} id="clear-chat-btn">
            🗑️ Clear conversation
          </button>
        )}
      </aside>

      {/* ── Main chat area ─────────────────────────────────────────────────── */}
      <main className={styles.main}>
        {/* Header */}
        <header className={styles.header}>
          <div className={styles.headerLeft}>
            <h2 className={styles.headerTitle}>Document Q&amp;A</h2>
            <span className={styles.headerSub}>
              {messages.length > 0
                ? `${Math.ceil(messages.length / 2)} turn${messages.length > 2 ? 's' : ''}`
                : 'New conversation'}
            </span>
          </div>
          {loading && (
            <div className={styles.thinkingBadge}>
              <span className="spinner" />
              <span>Thinking…</span>
            </div>
          )}
        </header>

        {/* Messages */}
        <ChatWindow messages={messages} />

        {/* Input */}
        <InputBar onSend={handleSend} disabled={loading || status === 'error'} />
      </main>
    </div>
  );
}

const FEATURES = [
  { icon: '🔍', label: 'Semantic search' },
  { icon: '🏆', label: 'CrossEncoder reranking' },
  { icon: '✅', label: 'Fact verification' },
  { icon: '💬', label: 'Multi-turn memory' },
  { icon: '⚡', label: 'Semantic caching' },
  { icon: '📄', label: 'PDF · DOCX · MD · HTML' },
];
