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
            <span className="doc-name">{doc.title ?? doc.filename}</span>
            <span className={`status-badge status-${doc.status}`}>{doc.status}</span>
            {doc.status === "ready" && (
              <span className={`version-badge ${doc.is_latest ? "version-latest" : "version-superseded"}`}>
                {doc.is_latest ? "Latest" : "Superseded"}
                {doc.version && ` (v${doc.version})`}
              </span>
            )}
            {doc.error_message && <span className="doc-error">{doc.error_message}</span>}
          </li>
        ))}
      </ul>
    </div>
  );
}
