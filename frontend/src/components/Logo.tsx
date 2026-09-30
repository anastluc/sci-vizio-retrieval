export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill="var(--accent)" />
      <rect x="7" y="17" width="4" height="8" rx="1" fill="#fff" />
      <rect x="14" y="11" width="4" height="14" rx="1" fill="#fff" />
      <rect x="21" y="7" width="4" height="18" rx="1" fill="#fff" />
    </svg>
  );
}
