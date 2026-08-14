export type CurvePoint = {
  generation: number;
  value: number;
  std?: number;
  averageTurns?: number;
};

export type CurveSeries = {
  id: string;
  label: string;
  kind: "train" | "eval" | "final" | "baseline";
  points: CurvePoint[];
  mask?: number;
  configId?: string;
  method?: string;
  capabilityCurve?: boolean;
};

export type ReactStep = {
  turn: number;
  assistant: string;
  observation: string;
  action?: { name?: string; arguments?: { command?: string } };
};

export type CaseCheckpoint = {
  generation: number;
  score: number;
  prediction: string;
  terminationReason: string;
  steps: ReactStep[];
  anls?: number;
  acc?: number;
};

export type TaskPayload = {
  metadata: {
    task: string;
    title?: string;
    method: string;
    metric?: string;
    sourceFiles: string[];
    note?: string;
  };
  configurations: Record<string, unknown>[];
  curves: CurveSeries[];
  checkpoints: Record<string, unknown>[];
  cases: Record<string, unknown>[];
  finalResults: Record<string, string | number>[];
};
