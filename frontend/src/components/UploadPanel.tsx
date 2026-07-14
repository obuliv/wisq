import { useRef, useState } from "react";
import { uploadDocument } from "../api/client";

interface Props {
  onUploaded: () => void;
}

export function UploadPanel({ onUploaded }: Props) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function handleUpload() {
    const file = fileInputRef.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setStatus(null);
    try {
      await uploadDocument(file);
      setStatus(`Uploaded ${file.name}`);
      if (fileInputRef.current) fileInputRef.current.value = "";
      onUploaded();
    } catch (err) {
      setStatus(`Upload failed: ${(err as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2>Upload document</h2>
      <input ref={fileInputRef} type="file" accept=".docx" />
      <button onClick={handleUpload} disabled={busy}>
        {busy ? "Uploading..." : "Upload"}
      </button>
      {status && <p className="status-line">{status}</p>}
    </div>
  );
}
