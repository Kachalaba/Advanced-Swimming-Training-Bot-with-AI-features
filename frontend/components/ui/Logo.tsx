export function Logo({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none">
      <defs>
        <linearGradient id="logo-grad" x1="0" y1="0" x2="32" y2="32">
          <stop offset="0%" stopColor="#22D3EE" />
          <stop offset="100%" stopColor="#0891B2" />
        </linearGradient>
      </defs>
      <path
        d="M4 22 L10 10 L16 18 L22 6 L28 22"
        stroke="url(#logo-grad)"
        strokeWidth="2.5"
        strokeLinecap="round"
        strokeLinejoin="round"
        fill="none"
      />
      <circle cx="28" cy="22" r="2.5" fill="#22D3EE" />
    </svg>
  );
}
