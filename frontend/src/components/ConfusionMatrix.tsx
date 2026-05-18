"use client";

import { useEffect, useRef } from "react";
import * as d3 from "d3";

const SIZE = 340;
const M = { top: 30, right: 20, bottom: 55, left: 55 };

export default function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !matrix) return;

    const innerSize = SIZE - M.left - M.right;
    const cellSize = innerSize / 10;
    const digits = d3.range(10);

    d3.select(svgRef.current).selectAll("*").remove();

    const svg = d3
      .select(svgRef.current)
      .append("g")
      .attr("transform", `translate(${M.left},${M.top})`);

    const maxVal = d3.max(matrix.flat()) ?? 1;
    const colorScale = d3.scaleSequential(d3.interpolateBlues).domain([0, maxVal]);

    matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        svg
          .append("rect")
          .attr("x", j * cellSize)
          .attr("y", i * cellSize)
          .attr("width", cellSize - 1)
          .attr("height", cellSize - 1)
          .attr("fill", colorScale(val))
          .attr("rx", 2);

        if (val > 0) {
          svg
            .append("text")
            .attr("x", j * cellSize + cellSize / 2)
            .attr("y", i * cellSize + cellSize / 2 + 3.5)
            .attr("text-anchor", "middle")
            .attr("font-size", 9)
            .attr("fill", val > maxVal * 0.55 ? "white" : "#1f2937")
            .text(val);
        }
      });
    });

    // Diagonal highlight (correct predictions)
    digits.forEach((d) => {
      svg
        .append("rect")
        .attr("x", d * cellSize)
        .attr("y", d * cellSize)
        .attr("width", cellSize - 1)
        .attr("height", cellSize - 1)
        .attr("fill", "none")
        .attr("stroke", "#f59e0b")
        .attr("stroke-width", 1.5)
        .attr("rx", 2);
    });

    const bandScale = d3.scaleBand<number>().domain(digits).range([0, innerSize]);

    svg
      .append("g")
      .call(d3.axisTop(bandScale).tickFormat((d) => String(d)))
      .call((g) => g.select(".domain").remove())
      .call((g) => g.selectAll("text").attr("fill", "#9ca3af").attr("font-size", 10))
      .call((g) => g.selectAll("line").remove());

    svg
      .append("g")
      .call(d3.axisLeft(bandScale).tickFormat((d) => String(d)))
      .call((g) => g.select(".domain").remove())
      .call((g) => g.selectAll("text").attr("fill", "#9ca3af").attr("font-size", 10))
      .call((g) => g.selectAll("line").remove());

    svg
      .append("text")
      .attr("x", innerSize / 2)
      .attr("y", -18)
      .attr("text-anchor", "middle")
      .attr("fill", "#6b7280")
      .attr("font-size", 11)
      .text("Predicted");

    svg
      .append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -innerSize / 2)
      .attr("y", -42)
      .attr("text-anchor", "middle")
      .attr("fill", "#6b7280")
      .attr("font-size", 11)
      .text("True Label");
  }, [matrix]);

  return (
    <svg
      ref={svgRef}
      width={SIZE}
      height={SIZE}
      viewBox={`0 0 ${SIZE} ${SIZE}`}
      className="w-full max-w-sm mx-auto"
    />
  );
}
