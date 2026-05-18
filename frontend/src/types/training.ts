export interface TrainingConfig {
  hidden_size: number;
  epochs: number;
  learning_rate: number;
  batch_size: number;
  training_size: number;
}

export interface LayerGradientNorms {
  weights: number;
  biases: number;
}

export interface GradientNorms {
  dense1: LayerGradientNorms;
  dense2: LayerGradientNorms;
}

export interface WeightLayerStats {
  mean: number;
  std: number;
  hist_counts: number[];
  hist_edges: number[];
  min: number;
  max: number;
}

export interface WeightStats {
  dense1: WeightLayerStats;
  dense2: WeightLayerStats;
}

export interface BiasLayerStats {
  values: number[];
  mean: number;
  std: number;
}

export interface BiasStats {
  dense1: BiasLayerStats;
  dense2: BiasLayerStats;
}

export interface ActivationLayerStats {
  mean: number;
  std: number;
  sparsity: number;
}

export interface ActivationStats {
  dense1: ActivationLayerStats;
}

export interface EpochEvent {
  type: "epoch";
  epoch: number;
  loss: number;
  accuracy: number;
  gradient_norms: GradientNorms;
  weight_stats: WeightStats;
  bias_stats: BiasStats;
  activation_stats: ActivationStats;
}

export interface MisclassifiedDigit {
  pixels: number[];
  true_label: number;
  predicted_label: number;
  confidence: number;
}

export interface CompleteEvent {
  type: "complete";
  test_results: { loss: number; accuracy: number };
  confusion_matrix: number[][];
  misclassified: MisclassifiedDigit[];
}

export type StreamEvent = EpochEvent | CompleteEvent;

export interface ChartPoint {
  epoch: number;
  loss: number;
  accuracy: number;
  dense1WeightGradNorm: number;
  dense1BiasGradNorm: number;
  dense2WeightGradNorm: number;
  dense2BiasGradNorm: number;
  activationMean: number;
  activationSparsity: number;
}

export interface TrainingState {
  isTraining: boolean;
  isComplete: boolean;
  error: string | null;
  chartPoints: ChartPoint[];
  latestWeightStats: WeightStats | null;
  latestBiasStats: BiasStats | null;
  confusionMatrix: number[][] | null;
  misclassified: MisclassifiedDigit[];
  testResults: { loss: number; accuracy: number } | null;
}
