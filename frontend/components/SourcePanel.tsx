'use client';
// components/SourcePanel.tsx — collapsible source citations

import { useState } from 'react';
import { Source } from '@/lib/api';
import styles from './SourcePanel.module.css';

interface Props { sources: Source[]; }

export default function SourcePanel({ sources }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <div className={styles.container}>
      <button
        className={styles.toggle}
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
      >
        <span className={styles.icon}>📄</span>
        {sources.length} source{sources.length !== 1 ? 's' : ''}
        <span className={`${styles.chevron} ${open ? styles.chevronOpen : ''}`}>›</span>
      </button>

      {open && (
        <div className={styles.panel}>
          {sources.map((src, i) => (
            <div key={i} className={styles.sourceItem}>
              <span className={styles.fileIcon}>📑</span>
              <div className={styles.sourceInfo}>
                <span className={styles.filename}>{src.filename}</span>
                {src.page !== null && (
                  <span className={styles.page}>page {src.page}</span>
                )}
              </div>
              {src.score !== null && (
                <div className={styles.scoreBar}>
                  <div
                    className={styles.scoreBarFill}
                    style={{ width: `${Math.min(src.score, 100)}%` }}
                  />
                  <span className={styles.scoreLabel}>{src.score.toFixed(1)}</span>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
