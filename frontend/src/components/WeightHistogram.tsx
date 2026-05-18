"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";
import type { WeightLayerStats } from "@/types/training";

interface Props {
  stats: WeightLayerStats;
  layerName: string;
  color?: string;
}

const W = 320;
const H = 140;
const M = { top: 10, right: 16, bottom: 28, left: 36 };

export default function WeightHistogram({ stats, layerName, color = "#6366f1" }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !stats) return;

    const innerW = W - M.left - M.right;
    const innerH = H - M.top - M.bottom;

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .append("g")
      .attr("transform", `translate(${M.left},${M.top})`);

    const xScale = d3
      .scaleLinear()
      .domain([stats.hist_edges[0], stats.hist_edges[stats.hist_edges.length - 1]])
      .range([0, innerW]);

    const yScale = d3
      .scaleLinear()
      .domain([0, d3.max(stats.hist_counts) ?? 1])
      .range([innerH, 0]);

    stats.hist_counts.forEach((count, i) => {
      const x0 = xScale(stats.hist_edges[i]);
      const x1 = xScale(stats.hist_edges[i + 1]);
      svg
        .append("rect")
        .attr("x", x0)
        .attr("y", yScale(count))
        .attr("width", Math.max(0, x1 - x0 - 0.5))
        .attr("height", innerH - yScale(count))
        .attr("fill", color)
        .attr("opacity", 0.8);
    });

    svg
      .append("line")
      .attr("x1", xScale(stats.mean))
      .attr("x2", xScale(stats.mean))
      .attr("y1", 0)
      .attr("y2", innerH)
      .attr("stroke", "#f59e0b")
      .attr("stroke-width", 1.5)
      .attr("stroke-dasharray", "3 2");

    svg
      .append("g")
      .attr("transform", `translate(0,${innerH})`)
      .call(d3.axisBottom(xScale).ticks(5).tickSize(3))
      .call((g) => g.select(".domain").attr("stroke", "#374151"))
      .call((g) => g.selectAll("text").attr("fill", "#9ca3af").attr("font-size", 9))
      .call((g) => g.selectAll("line").attr("stroke", "#374151"));

    svg
      .append("g")
      .call(d3.axisLeft(yScale).ticks(3).tickSize(3))
      .call((g) => g.select(".domain").attr("stroke", "#374151"))
      .call((g) => g.selectAll("text").attr("fill", "#9ca3af").attr("font-size", 9))
      .call((g) => g.selectAll("line").attr("stroke", "#374151"));
  }, [stats, color]);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex justify-between items-center">
        <span className="text-xs text-gray-400">{layerName}</span>
        <span className="text-xs font-mono text-gray-500">
          μ={stats.mean.toFixed(4)} σ={stats.std.toFixed(4)}
        </span>
      </div>
      <svg ref={svgRef} width={W} height={H} className="w-full" viewBox={`0 0 ${W} ${H}`} />
    </div>
  );
}
