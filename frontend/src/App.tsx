import { useEffect, useState } from "react";
import { type DocumentOut, listDocuments } from "./api/client";
import { ChatWindow } from "./components/ChatWindow";
import { DocumentList } from "./components/DocumentList";
import { UploadPanel } from "./components/UploadPanel";

export function App() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [documentsLoading, setDocumentsLoading] = useState(true);
  const [documentsError, setDocumentsError] = useState<string | null>(null);

  async function refreshDocuments() {
    try {
      const docs = await listDocuments();
      setDocuments(docs);
      setDocumentsError(null);
    } catch (err) {
      setDocumentsError((err as Error).message);
    } finally {
      setDocumentsLoading(false);
    }
  }

  useEffect(() => {
    refreshDocuments();
    const interval = setInterval(refreshDocuments, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      <header>
        <svg className="header-icon" width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M4 4h13a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H8l-4 3V4Z"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinejoin="round"
          />
        </svg>
        <h1>Wisq Q&amp;A</h1>
      </header>
      <main>
        <div className="sidebar">
          <UploadPanel onUploaded={refreshDocuments} />
          <DocumentList documents={documents} loading={documentsLoading} error={documentsError} />
        </div>
        <ChatWindow />
      </main>
    </div>
  );
}
