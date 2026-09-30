import { api } from "../api/client";
import type { DocumentItem } from "../api/types";

export async function sha256Hex(data: Blob): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", await data.arrayBuffer());
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

function putWithProgress(
  url: string,
  method: string,
  headers: Record<string, string>,
  file: Blob,
  onProgress: (fraction: number) => void,
): Promise<void> {
  // fetch() has no upload progress events, so use XMLHttpRequest.
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open(method, url);
    for (const [k, v] of Object.entries(headers)) xhr.setRequestHeader(k, v);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () =>
      xhr.status >= 200 && xhr.status < 300 ? resolve() : reject(new Error(`Upload failed (${xhr.status})`));
    xhr.onerror = () => reject(new Error("Upload failed. Check your connection."));
    xhr.send(file);
  });
}

/**
 * Upload a PDF: reserve it with the API, send the bytes straight to object
 * storage through the signed URL, then ask the API to start processing.
 */
export async function uploadPdf(
  file: File,
  collectionId: string | null,
  onProgress: (fraction: number) => void = () => {},
): Promise<DocumentItem> {
  const sha256 = await sha256Hex(file);
  const { document, upload } = await api.startUpload({
    filename: file.name,
    size_bytes: file.size,
    sha256,
    collection_id: collectionId,
  });
  try {
    await putWithProgress(upload.url, upload.method, upload.headers, file, onProgress);
  } catch (error) {
    // Release the reserved slot so it doesn't count against the quota.
    await api.deleteDocument(document.id).catch(() => undefined);
    throw error;
  }
  return api.completeUpload(document.id);
}
