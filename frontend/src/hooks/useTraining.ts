"use client";

import { useState, useCallback, useRef } from "react";
import type {
  TrainingConfig,
  TrainingState,
  ChartPoint,
  EpochEvent,
  CompleteEvent,
  StreamEvent,
} from "@/types/training";

const INITIAL_STATE: TrainingState = {
  isTraining: false,
  isComplete: false,
  error: null,
  chartPoints: [],
  latestWeightStats: null,
  latestBiasStats: null,
  confusionMatrix: null,
  misclassified: [],
  testResults: null,
};

export function useTraining() {
  const [state, setState] = useState<TrainingState>(INITIAL_STATE);
  const abortRef = useRef<AbortController | null>(null);

  const handleEvent = useCallback((event: StreamEvent) => {
    if (event.type === "epoch") {
      const e = event as EpochEvent;
      const point: ChartPoint = {
        epoch: e.epoch,
        loss: e.loss,
        accuracy: e.accuracy,
        dense1WeightGradNorm: e.gradient_norms.dense1.weights,
        dense1BiasGradNorm: e.gradient_norms.dense1.biases,
        dense2WeightGradNorm: e.gradient_norms.dense2.weights,
        dense2BiasGradNorm: e.gradient_norms.dense2.biases,
        activationMean: e.activation_stats.dense1.mean,
        activationSparsity: e.activation_stats.dense1.sparsity,
      };
      setState((prev) => ({
        ...prev,
        chartPoints: [...prev.chartPoints, point],
        latestWeightStats: e.weight_stats,
        latestBiasStats: e.bias_stats,
      }));
    } else if (event.type === "complete") {
      const e = event as CompleteEvent;
      setState((prev) => ({
        ...prev,
        isTraining: false,
        isComplete: true,
        confusionMatrix: e.confusion_matrix,
        misclassified: e.misclassified,
        testResults: e.test_results,
      }));
    }
  }, []);

  const startTraining = useCallback(
    async (config: TrainingConfig) => {
      if (abortRef.current) abortRef.current.abort();
      const abort = new AbortController();
      abortRef.current = abort;

      setState({ ...INITIAL_STATE, isTraining: true });

      try {
        const response = await fetch("http://localhost:8000/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(config),
          signal: abort.signal,
        });

        if (!response.ok) throw new Error(`Server error: ${response.status}`);
        if (!response.body) throw new Error("No response body");

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const messages = buffer.split("\n\n");
          buffer = messages.pop() ?? "";

          for (const message of messages) {
            const line = message.trim();
            if (!line.startsWith("data: ")) continue;
            const event: StreamEvent = JSON.parse(line.slice(6));
            handleEvent(event);
          }
        }
      } catch (err) {
        if ((err as Error).name === "AbortError") return;
        setState((prev) => ({
          ...prev,
          isTraining: false,
          error: (err as Error).message,
        }));
      }
    },
    [handleEvent]
  );

  const stopTraining = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, isTraining: false }));
  }, []);

  const reset = useCallback(() => {
    abortRef.current?.abort();
    setState(INITIAL_STATE);
  }, []);

  return { state, startTraining, stopTraining, reset };
}
