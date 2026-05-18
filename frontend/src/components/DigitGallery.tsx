import DigitCanvas from "./DigitCanvas";
import type { MisclassifiedDigit } from "@/types/training";

export default function DigitGallery({ digits }: { digits: MisclassifiedDigit[] }) {
  if (digits.length === 0) return null;

  return (
    <div className="grid grid-cols-4 gap-4">
      {digits.map((d, i) => (
        <div
          key={i}
          className="flex flex-col items-center gap-1 bg-gray-800 rounded-lg p-2"
        >
          <DigitCanvas pixels={d.pixels} scale={3} />
          <p className="text-xs text-red-400 font-mono">pred: {d.predicted_label}</p>
          <p className="text-xs text-green-400 font-mono">true: {d.true_label}</p>
          <p className="text-xs text-gray-500 font-mono">
            {(d.confidence * 100).toFixed(1)}%
          </p>
        </div>
      ))}
    </div>
  );
}
