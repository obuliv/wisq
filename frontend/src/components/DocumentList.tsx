import type { DocumentOut } from "../api/client";

interface Props {
  documents: DocumentOut[];
  loading: boolean;
  error: string | null;
}

export function DocumentList({ documents, loading, error }: Props) {
  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Documents</h2>
      </div>

      {error && (
        <p className="list-error-banner" role="alert">
          Couldn&apos;t refresh documents: {error}
        </p>
      )}

      {loading && documents.length === 0 ? (
        <div className="doc-list-skeleton" aria-hidden="true">
          <div className="skeleton" />
          <div className="skeleton" />
          <div className="skeleton" />
        </div>
      ) : documents.length === 0 ? (
        <div className="empty-state">
          <p className="muted">No documents uploaded yet.</p>
        </div>
      ) : (
        <ul className="doc-list" aria-label="Uploaded documents">
          {documents.map((doc) => (
            <li key={doc.id}>
              <span className="doc-name">{doc.title ?? doc.filename}</span>
              <span className="badge-row">
                <span className={`status-badge status-${doc.status}`}>{doc.status}</span>
                {doc.status === "ready" && (
                  <span className={`version-badge ${doc.is_latest ? "version-latest" : "version-superseded"}`}>
                    {doc.is_latest ? "Latest" : "Superseded"}
                    {doc.version && ` (v${doc.version})`}
                  </span>
                )}
              </span>
              {doc.error_message && <span className="doc-error">{doc.error_message}</span>}
            </li>
          ))}
        </ul>
      )}
      <span className="sr-only" aria-live="polite">
        {loading ? "Loading documents…" : `${documents.length} document${documents.length === 1 ? "" : "s"} loaded`}
      </span>
    </div>
  );
}
