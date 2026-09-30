import { api } from "../api/client";
import { sha256Hex, uploadPdf } from "./upload";

class FakeXHR {
  static status = 200;
  status = 0;
  upload: { onprogress?: (e: ProgressEvent) => void } = {};
  onload?: () => void;
  onerror?: () => void;
  open = vi.fn();
  setRequestHeader = vi.fn();
  send() {
    this.upload.onprogress?.({ lengthComputable: true, loaded: 5, total: 10 } as ProgressEvent);
    this.status = FakeXHR.status;
    this.onload?.();
  }
}

beforeEach(() => vi.stubGlobal("XMLHttpRequest", FakeXHR));
afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
  FakeXHR.status = 200;
});

it("hashes files with SHA-256", async () => {
  expect(await sha256Hex(new Blob(["abc"]))).toBe(
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
  );
});

it("reserves, uploads to the signed URL and completes", async () => {
  const doc = { id: "d1" } as never;
  const start = vi.spyOn(api, "startUpload").mockResolvedValue({
    document: doc,
    upload: { url: "https://bucket/signed", method: "PUT", headers: { "Content-Type": "application/pdf" } },
  });
  const complete = vi.spyOn(api, "completeUpload").mockResolvedValue(doc);
  const progress = vi.fn();

  const file = new File(["%PDF-1.7"], "paper.pdf", { type: "application/pdf" });
  await uploadPdf(file, "c1", progress);

  expect(start).toHaveBeenCalledWith(expect.objectContaining({ filename: "paper.pdf", size_bytes: 8, collection_id: "c1" }));
  expect(progress).toHaveBeenCalledWith(0.5);
  expect(complete).toHaveBeenCalledWith("d1");
});

it("releases the reserved slot when the upload fails", async () => {
  FakeXHR.status = 403;
  vi.spyOn(api, "startUpload").mockResolvedValue({
    document: { id: "d1" } as never,
    upload: { url: "https://bucket/signed", method: "PUT", headers: {} },
  });
  const remove = vi.spyOn(api, "deleteDocument").mockResolvedValue(undefined);
  const complete = vi.spyOn(api, "completeUpload");

  await expect(uploadPdf(new File(["x"], "a.pdf"), null)).rejects.toThrow("Upload failed (403)");
  expect(remove).toHaveBeenCalledWith("d1");
  expect(complete).not.toHaveBeenCalled();
});
