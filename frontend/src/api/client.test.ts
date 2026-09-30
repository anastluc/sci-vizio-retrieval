import { ApiError, api, request, setTokenProvider } from "./client";

function mockFetch(status: number, body: unknown) {
  const fn = vi.fn().mockResolvedValue(
    new Response(body === undefined ? null : JSON.stringify(body), { status }),
  );
  vi.stubGlobal("fetch", fn);
  return fn;
}

afterEach(() => {
  vi.unstubAllGlobals();
  setTokenProvider(async () => null);
});

it("sends the Firebase token and JSON body", async () => {
  setTokenProvider(async () => "id-token");
  const fetch = mockFetch(200, { id: "c1", name: "Thesis" });

  await api.createCollection("Thesis");

  const [url, init] = fetch.mock.calls[0];
  expect(url).toBe("http://localhost:8000/api/collections");
  expect(init.method).toBe("POST");
  expect(new Headers(init.headers).get("Authorization")).toBe("Bearer id-token");
  expect(init.body).toBe(JSON.stringify({ name: "Thesis" }));
});

it("surfaces structured quota errors", async () => {
  mockFetch(402, { detail: { code: "document_limit", message: "The Free plan allows 10 PDFs." } });
  const error = await request("/documents/uploads", { method: "POST" }).catch((e) => e);
  expect(error).toBeInstanceOf(ApiError);
  expect(error).toMatchObject({ status: 402, code: "document_limit", message: "The Free plan allows 10 PDFs." });
});

it("surfaces validation errors", async () => {
  mockFetch(422, { detail: [{ msg: "Field required" }] });
  await expect(request("/search")).rejects.toThrow("Field required");
});

it("handles empty 204 responses", async () => {
  mockFetch(204, undefined);
  await expect(api.deleteDocument("d1")).resolves.toBeUndefined();
});

it("reports network failures clearly", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
  await expect(api.me()).rejects.toMatchObject({ status: 0, message: expect.stringContaining("Can't reach") });
});
