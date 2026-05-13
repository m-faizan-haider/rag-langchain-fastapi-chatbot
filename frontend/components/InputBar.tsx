'use client';
// components/InputBar.tsx — message input with send button

import { useState, useRef, KeyboardEvent } from 'react';
import styles from './InputBar.module.css';

interface Props {
  onSend:    (text: string) => void;
  disabled?: boolean;
}

export default function InputBar({ onSend, disabled = false }: Props) {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    const trimmed = value.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setValue('');
    // Reset height
    if (textareaRef.current) textareaRef.current.style.height = 'auto';
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = () => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  };

  return (
    <div className={styles.container}>
      <div className={`${styles.inputWrapper} ${disabled ? styles.inputDisabled : ''}`}>
        <textarea
          ref={textareaRef}
          id="chat-input"
          className={styles.textarea}
          value={value}
          onChange={(e) => { setValue(e.target.value); handleInput(); }}
          onKeyDown={handleKeyDown}
          placeholder="Ask anything about your documents…"
          rows={1}
          disabled={disabled}
          aria-label="Chat input"
        />

        <button
          id="send-button"
          className={`${styles.sendBtn} ${(!value.trim() || disabled) ? styles.sendBtnDisabled : ''}`}
          onClick={handleSend}
          disabled={!value.trim() || disabled}
          aria-label="Send message"
        >
          {disabled ? (
            <span className="spinner" />
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
              <line x1="22" y1="2" x2="11" y2="13"/>
              <polygon points="22 2 15 22 11 13 2 9 22 2"/>
            </svg>
          )}
        </button>
      </div>
      <p className={styles.hint}>Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line</p>
    </div>
  );
}
