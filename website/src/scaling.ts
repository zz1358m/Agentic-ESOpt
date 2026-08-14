export type ModelSize = "4B" | "9B";
export type EsPopulationSize = 8 | 16;

export type ScaleResult = {
  model: ModelSize;
  population: EsPopulationSize;
  best: number;
  final: number;
  bestRelative?: number;
  finalRelative?: number;
};

export const MODEL_SIZES: ModelSize[] = ["4B", "9B"];
export const ES_POPULATION_SIZES: EsPopulationSize[] = [8, 16];

type ScalingKey = `${ModelSize}-${EsPopulationSize}`;
export type ScalingMatrix = Record<ScalingKey, ScaleResult>;

export const scalingKey = (model: ModelSize, population: EsPopulationSize): ScalingKey => `${model}-${population}`;

export function buildScalingMatrix(rows: ScaleResult[]): ScalingMatrix | null {
  if (rows.length !== MODEL_SIZES.length * ES_POPULATION_SIZES.length) return null;
  const matrix: Partial<ScalingMatrix> = {};
  for (const row of rows) {
    if (!MODEL_SIZES.includes(row.model) || !ES_POPULATION_SIZES.includes(row.population)) return null;
    if (!Number.isFinite(row.best) || !Number.isFinite(row.final)) return null;
    const key = scalingKey(row.model, row.population);
    if (matrix[key]) return null;
    matrix[key] = row;
  }
  return MODEL_SIZES.every((model) => ES_POPULATION_SIZES.every((population) => matrix[scalingKey(model, population)]))
    ? matrix as ScalingMatrix
    : null;
}
