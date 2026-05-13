import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title:       'RAG Chat — Document Q&A',
  description: 'Production-grade Retrieval-Augmented Generation chatbot. Ask questions about your documents powered by LangChain, FAISS, and AI.',
  keywords:    ['RAG', 'LangChain', 'document QA', 'AI chatbot', 'retrieval augmented generation'],
  openGraph: {
    title:       'RAG Chat — Document Q&A',
    description: 'Ask questions about your documents using AI.',
    type:        'website',
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
