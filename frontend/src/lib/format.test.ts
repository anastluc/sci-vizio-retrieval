import { formatBytes, formatPrice, pluralize } from "./format";

describe("formatBytes", () => {
  it.each([
    [0, "0 B"],
    [1023, "1023 B"],
    [1536, "1.5 KB"],
    [20 * 1024 * 1024, "20 MB"],
    [3.25 * 1024 ** 3, "3.3 GB"],
  ])("%d -> %s", (bytes, expected) => expect(formatBytes(bytes)).toBe(expected));
});

it("formats prices", () => {
  expect(formatPrice(0)).toBe("$0");
  expect(formatPrice(9)).toBe("$9");
  expect(formatPrice(9.5)).toBe("$9.50");
});

it("pluralizes", () => {
  expect(pluralize(1, "figure")).toBe("1 figure");
  expect(pluralize(3, "figure")).toBe("3 figures");
});
