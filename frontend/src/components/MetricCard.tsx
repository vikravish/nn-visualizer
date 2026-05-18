interface Props {
  label: string;
  value: number | null;
  format?: (v: number) => string;
  subtitle?: string;
  highlight?: boolean;
}

export default function MetricCard({ label, value, format, subtitle, highlight }: Props) {
  const displayValue =
    value === null
      ? "—"
      : format
      ? format(value)
      : value.toFixed(4);

  return (
    <div
      className={`rounded-xl p-4 flex flex-col gap-1 ${
        highlight ? "bg-indigo-900/40 border border-indigo-500/30" : "bg-gray-900"
      }`}
    >
      <span className="text-xs text-gray-400 uppercase tracking-wider">{label}</span>
      <span className="text-2xl font-mono font-bold text-white">{displayValue}</span>
      {subtitle && <span className="text-xs text-gray-500">{subtitle}</span>}
    </div>
  );
}
