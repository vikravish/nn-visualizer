import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ChartPoint } from "@/types/training";

export default function GradientNormChart({ data }: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="epoch"
          stroke="#6b7280"
          tick={{ fontSize: 11 }}
          label={{ value: "Epoch", position: "insideBottom", offset: -10, fill: "#6b7280", fontSize: 11 }}
        />
        <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} width={55} />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
          labelStyle={{ color: "#9ca3af" }}
          formatter={(v) => (typeof v === "number" ? v.toFixed(5) : v)}
        />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
        <Line type="monotone" dataKey="dense1WeightGradNorm" name="L1 weights" stroke="#6366f1" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="dense1BiasGradNorm" name="L1 biases" stroke="#818cf8" strokeWidth={1.5} dot={false} isAnimationActive={false} strokeDasharray="4 2" />
        <Line type="monotone" dataKey="dense2WeightGradNorm" name="L2 weights" stroke="#f59e0b" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="dense2BiasGradNorm" name="L2 biases" stroke="#fcd34d" strokeWidth={1.5} dot={false} isAnimationActive={false} strokeDasharray="4 2" />
      </LineChart>
    </ResponsiveContainer>
  );
}
