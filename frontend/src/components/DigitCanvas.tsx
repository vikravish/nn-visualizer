"use client";

import { useEffect, useRef } from "react";

interface Props {
  pixels: number[];
  scale?: number;
}

export default function DigitCanvas({ pixels, scale = 3 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imageData = ctx.createImageData(28, 28);
    pixels.forEach((v, i) => {
      const val = Math.round(v * 255);
      imageData.data[i * 4 + 0] = val;
      imageData.data[i * 4 + 1] = val;
      imageData.data[i * 4 + 2] = val;
      imageData.data[i * 4 + 3] = 255;
    });
    ctx.putImageData(imageData, 0, 0);
  }, [pixels]);

  return (
    <canvas
      ref={canvasRef}
      width={28}
      height={28}
      style={{
        width: 28 * scale,
        height: 28 * scale,
        imageRendering: "pixelated",
      }}
      className="rounded"
    />
  );
}
