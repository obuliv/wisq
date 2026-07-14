import { useEffect, useState } from "react";
import { type DocumentOut, listDocuments } from "./api/client";
import { ChatWindow } from "./components/ChatWindow";
import { DocumentList } from "./components/DocumentList";
import { UploadPanel } from "./components/UploadPanel";

export function App() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);

  async function refreshDocuments() {
    setDocuments(await listDocuments());
  }

  useEffect(() => {
    refreshDocuments();
    const interval = setInterval(refreshDocuments, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app">
      <header>
        <h1>Wisq Q&amp;A</h1>
      </header>
      <main>
        <div className="sidebar">
          <UploadPanel onUploaded={refreshDocuments} />
          <DocumentList documents={documents} />
        </div>
        <ChatWindow />
      </main>
    </div>
  );
}
