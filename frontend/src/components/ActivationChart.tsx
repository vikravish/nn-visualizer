import {
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ChartPoint } from "@/types/training";

export default function ActivationChart({ data }: { data: ChartPoint[] }) {
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 5, right: 40, bottom: 20, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
        <XAxis
          dataKey="epoch"
          stroke="#6b7280"
          tick={{ fontSize: 11 }}
          label={{ value: "Epoch", position: "insideBottom", offset: -10, fill: "#6b7280", fontSize: 11 }}
        />
        <YAxis yAxisId="mean" stroke="#10b981" tick={{ fontSize: 11 }} width={45} />
        <YAxis
          yAxisId="sparsity"
          orientation="right"
          stroke="#f59e0b"
          tick={{ fontSize: 11 }}
          width={45}
          tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
          domain={[0, 1]}
        />
        <Tooltip
          contentStyle={{ background: "#111827", border: "1px solid #374151", borderRadius: 8 }}
          labelStyle={{ color: "#9ca3af" }}
          formatter={(v, name) =>
            typeof v === "number"
              ? name === "Sparsity"
                ? `${(v * 100).toFixed(1)}%`
                : v.toFixed(4)
              : v
          }
        />
        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 8 }} />
        <Line
          yAxisId="mean"
          type="monotone"
          dataKey="activationMean"
          name="Mean activation"
          stroke="#10b981"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          yAxisId="sparsity"
          type="monotone"
          dataKey="activationSparsity"
          name="Sparsity"
          stroke="#f59e0b"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
