import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import type { ChartPoint } from "@/types/training";

export default function LossChart({ data }: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 10, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="epoch"
          stroke="#6b7280"
          tick={{ fontSize: 11 }}
          label={{ value: "Epoch", position: "insideBottom", offset: -10, fill: "#6b7280", fontSize: 11 }}
        />
        <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} width={50} />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
          labelStyle={{ color: "#9ca3af" }}
          itemStyle={{ color: "#ef4444" }}
          formatter={(v) => (typeof v === "number" ? v.toFixed(4) : v)}
        />
        <Line
          type="monotone"
          dataKey="loss"
          stroke="#ef4444"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
