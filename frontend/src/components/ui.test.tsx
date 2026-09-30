import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";

import { FacetGroup, StatusBadge, UsageMeter } from "./ui";

it("shows usage against the plan limit", () => {
  render(<UsageMeter label="PDFs" used={8} limit={10} />);
  expect(screen.getByText("8 / 10")).toBeInTheDocument();
  const bar = screen.getByRole("progressbar", { name: "PDFs" });
  expect(bar).toHaveAttribute("aria-valuenow", "8");
  expect(bar).toHaveClass("meter-high");
});

it("marks a full quota", () => {
  render(<UsageMeter label="PDFs" used={10} limit={10} />);
  expect(screen.getByRole("progressbar")).toHaveClass("meter-full");
});

it("labels document statuses", () => {
  render(<StatusBadge status="processing" />);
  expect(screen.getByText("Processing")).toBeInTheDocument();
});

it("toggles facet values", async () => {
  const onToggle = vi.fn();
  render(
    <FacetGroup
      title="Figure type"
      values={[{ value: "Bar chart", label: "Bar chart", count: 4 }, { value: "Diagram", label: "Diagram", count: 2 }]}
      selected={["Diagram"]}
      onToggle={onToggle}
    />,
  );
  expect(screen.getByRole("checkbox", { name: /Diagram/ })).toBeChecked();
  await userEvent.click(screen.getByRole("checkbox", { name: /Bar chart/ }));
  expect(onToggle).toHaveBeenCalledWith("Bar chart");
});
