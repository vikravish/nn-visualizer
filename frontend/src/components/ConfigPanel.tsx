"use client";

import { useState } from "react";
import type { TrainingConfig } from "@/types/training";

interface Props {
  onStart: (config: TrainingConfig) => void;
  onStop: () => void;
  isTraining: boolean;
}

export default function ConfigPanel({ onStart, onStop, isTraining }: Props) {
  const [config, setConfig] = useState<TrainingConfig>({
    hidden_size: 128,
    epochs: 20,
    learning_rate: 0.01,
    batch_size: 32,
    training_size: 10000,
  });

  const set = (key: keyof TrainingConfig, value: number) =>
    setConfig((prev) => ({ ...prev, [key]: value }));

  const sliders: {
    key: keyof TrainingConfig;
    label: string;
    min: number;
    max: number;
    step: number;
    format?: (v: number) => string;
  }[] = [
    { key: "hidden_size", label: "Hidden size", min: 32, max: 512, step: 32 },
    { key: "epochs", label: "Epochs", min: 1, max: 100, step: 1 },
    {
      key: "learning_rate",
      label: "Learning rate",
      min: -4,
      max: -1,
      step: 0.1,
      format: (v) => Math.pow(10, v).toExponential(2),
    },
    { key: "batch_size", label: "Batch size", min: 16, max: 256, step: 16 },
    {
      key: "training_size",
      label: "Training size",
      min: 1000,
      max: 60000,
      step: 1000,
      format: (v) => v.toLocaleString(),
    },
  ];

  const lrExponent = Math.log10(config.learning_rate);

  return (
    <div className="bg-gray-900 rounded-xl p-5 flex flex-col gap-4">
      <h2 className="text-sm font-semibold text-gray-400 uppercase tracking-wider">
        Config
      </h2>

      {sliders.map(({ key, label, min, max, step, format }) => {
        const isLR = key === "learning_rate";
        const sliderValue = isLR ? lrExponent : config[key];
        const displayValue = isLR
          ? config.learning_rate.toExponential(2)
          : format
          ? format(config[key] as number)
          : config[key];

        return (
          <div key={key} className="flex flex-col gap-1">
            <div className="flex justify-between items-center">
              <label className="text-xs text-gray-400">{label}</label>
              <span className="text-xs font-mono text-white">{displayValue}</span>
            </div>
            <input
              type="range"
              min={min}
              max={max}
              step={step}
              value={sliderValue as number}
              onChange={(e) => {
                const raw = parseFloat(e.target.value);
                set(key, isLR ? parseFloat(Math.pow(10, raw).toFixed(6)) : raw);
              }}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-indigo-500"
            />
          </div>
        );
      })}

      <button
        onClick={isTraining ? onStop : () => onStart(config)}
        className={`mt-2 w-full py-2.5 rounded-lg font-semibold text-sm transition-colors ${
          isTraining
            ? "bg-red-600 hover:bg-red-700 text-white"
            : "bg-indigo-600 hover:bg-indigo-700 text-white"
        }`}
      >
        {isTraining ? "Stop Training" : "Start Training"}
      </button>
    </div>
  );
}
