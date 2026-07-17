import { useRef, useState } from "react";
import { uploadDocument } from "../api/client";

interface Props {
  onUploaded: () => void;
}

export function UploadPanel({ onUploaded }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [statusKind, setStatusKind] = useState<"success" | "error" | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleUpload() {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setStatus(null);
    setStatusKind(null);
    try {
      await uploadDocument(file);
      setStatus(`Uploaded ${file.name}`);
      setStatusKind("success");
      if (fileInputRef.current) fileInputRef.current.value = "";
      onUploaded();
    } catch (err) {
      setStatus(`Upload failed: ${(err as Error).message}`);
      setStatusKind("error");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <div className="panel-header">
        <h2>Upload document</h2>
      </div>
      <div className="upload-dropzone">
        <label htmlFor="doc-upload-input" className="sr-only">
          Choose a .docx file to upload
        </label>
        <input id="doc-upload-input" ref={fileInputRef} type="file" accept=".docx" />
      </div>
      <button className="btn btn-primary" onClick={handleUpload} disabled={busy} aria-busy={busy}>
        {busy && <span className="spinner" aria-hidden="true" />}
        {busy ? "Uploading…" : "Upload"}
      </button>
      {status && (
        <p
          className={`status-line ${statusKind === "error" ? "status-line-error" : "status-line-success"}`}
          role="status"
        >
          {status}
        </p>
      )}
    </div>
  );
}
