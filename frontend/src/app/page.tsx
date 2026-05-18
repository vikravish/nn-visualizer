"use client";

import { useTraining } from "@/hooks/useTraining";
import ConfigPanel from "@/components/ConfigPanel";
import MetricCard from "@/components/MetricCard";
import LossChart from "@/components/LossChart";
import AccuracyChart from "@/components/AccuracyChart";
import GradientNormChart from "@/components/GradientNormChart";
import WeightHistogram from "@/components/WeightHistogram";
import ActivationChart from "@/components/ActivationChart";
import ConfusionMatrix from "@/components/ConfusionMatrix";
import DigitGallery from "@/components/DigitGallery";

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-gray-900 rounded-xl p-4">
      <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
        {title}
      </h2>
      {children}
    </div>
  );
}

export default function Home() {
  const { state, startTraining, stopTraining, reset } = useTraining();
  const latest = state.chartPoints[state.chartPoints.length - 1];
  const hasData = state.chartPoints.length > 0;

  return (
    <main className="min-h-screen bg-gray-950 text-white p-6 flex flex-col gap-5">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">NN Visualizer</h1>
          <p className="text-xs text-gray-500 mt-0.5">
            MNIST digit classifier · from-scratch NumPy MLP
          </p>
        </div>
        <div className="flex items-center gap-3">
          {state.isTraining && (
            <span className="flex items-center gap-1.5 text-xs text-indigo-400">
              <span className="w-2 h-2 rounded-full bg-indigo-400 animate-pulse" />
              Training — epoch {latest?.epoch ?? 0}
            </span>
          )}
          {state.isComplete && (
            <span className="text-xs text-green-400">Training complete</span>
          )}
          {state.error && (
            <span className="text-xs text-red-400">{state.error}</span>
          )}
          {(state.isComplete || state.error) && (
            <button
              onClick={reset}
              className="text-xs text-gray-400 hover:text-white underline"
            >
              Reset
            </button>
          )}
        </div>
      </div>

      {/* Top row: config + metric cards */}
      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-1">
          <ConfigPanel
            onStart={startTraining}
            onStop={stopTraining}
            isTraining={state.isTraining}
          />
        </div>
        <div className="col-span-3 grid grid-cols-2 grid-rows-2 gap-3">
          <MetricCard label="Training Loss" value={latest?.loss ?? null} />
          <MetricCard
            label="Training Accuracy"
            value={latest?.accuracy ?? null}
            format={(v) => `${(v * 100).toFixed(1)}%`}
          />
          <MetricCard
            label="Test Loss"
            value={state.testResults?.loss ?? null}
            highlight={state.isComplete}
            subtitle={state.isComplete ? "final" : undefined}
          />
          <MetricCard
            label="Test Accuracy"
            value={state.testResults?.accuracy ?? null}
            format={(v) => `${(v * 100).toFixed(1)}%`}
            highlight={state.isComplete}
            subtitle={state.isComplete ? "final" : undefined}
          />
        </div>
      </div>

      {/* Charts */}
      {hasData && (
        <>
          <div className="grid grid-cols-2 gap-4">
            <Section title="Loss">
              <LossChart data={state.chartPoints} />
            </Section>
            <Section title="Accuracy">
              <AccuracyChart data={state.chartPoints} />
            </Section>
          </div>

          <Section title="Gradient Norms">
            <GradientNormChart data={state.chartPoints} />
          </Section>

          <div className="grid grid-cols-2 gap-4">
            <Section title="Layer 1 Activations">
              <ActivationChart data={state.chartPoints} />
            </Section>
            {state.latestWeightStats && (
              <Section title="Weight Distributions">
                <div className="flex flex-col gap-4">
                  <WeightHistogram
                    stats={state.latestWeightStats.dense1}
                    layerName="Dense 1  (784 → hidden)"
                    color="#6366f1"
                  />
                  <WeightHistogram
                    stats={state.latestWeightStats.dense2}
                    layerName="Dense 2  (hidden → 10)"
                    color="#ec4899"
                  />
                </div>
              </Section>
            )}
          </div>

          {state.isComplete && state.confusionMatrix && (
            <div className="grid grid-cols-2 gap-4">
              <Section title="Confusion Matrix">
                <ConfusionMatrix matrix={state.confusionMatrix} />
              </Section>
              <Section
                title={`Misclassified Digits (${state.misclassified.length})`}
              >
                <DigitGallery digits={state.misclassified} />
              </Section>
            </div>
          )}
        </>
      )}

      {!hasData && !state.isTraining && (
        <div className="flex-1 flex items-center justify-center text-gray-600 text-sm">
          Configure and start training to see live charts
        </div>
      )}
    </main>
  );
}
