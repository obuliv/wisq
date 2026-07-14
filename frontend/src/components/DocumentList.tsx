import type { DocumentOut } from "../api/client";

interface Props {
  documents: DocumentOut[];
}

export function DocumentList({ documents }: Props) {
  return (
    <div className="panel">
      <h2>Documents</h2>
      {documents.length === 0 && <p className="muted">No documents uploaded yet.</p>}
      <ul className="doc-list">
        {documents.map((doc) => (
          <li key={doc.id}>
            <span className="doc-name">{doc.filename}</span>
            <span className={`status-badge status-${doc.status}`}>{doc.status}</span>
            {doc.error_message && <span className="doc-error">{doc.error_message}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
