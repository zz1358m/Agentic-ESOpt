import { useTaskData } from "../hooks/useTaskData";
import { buildScalingMatrix, ES_POPULATION_SIZES, MODEL_SIZES, scalingKey, type ScaleResult } from "../scaling";

export function ScalingSummary({ rows }: { rows: ScaleResult[] }) {
  const matrix = buildScalingMatrix(rows);
  if (!matrix) return <div className="empty-state">The complete scaling matrix is unavailable.</div>;
  const finalSensitivity = (model: typeof MODEL_SIZES[number]): number | null => {
    const at8 = matrix[scalingKey(model, 8)].final;
    const at16 = matrix[scalingKey(model, 16)];
    if (at16.finalRelative !== undefined) return at16.finalRelative;
    if (at8 === 0) return at16.final === 0 ? 0 : null;
    return (at16.final - at8) / at8 * 100;
  };
  const sensitivities = MODEL_SIZES.map((model) => ({ model, value: finalSensitivity(model) }));
  const highestSensitivity = Math.max(...sensitivities.flatMap(({ value }) => value === null ? [] : [value]));
  return <div className="scale-summary">
    <div className="scale-matrix scale-matrix--scaling" aria-label="Model-size and ES population scaling matrix">
      <span>Best / final test success</span>
      {MODEL_SIZES.flatMap((model) => ES_POPULATION_SIZES.map((population) => {
        const row = matrix[scalingKey(model, population)];
        return <div key={`${model}-${population}`}>
          <small>{model} · G={population}</small><strong>{row.final.toFixed(2)}% final</strong><em>best {row.best.toFixed(2)}%</em>
        </div>;
      }))}
    </div>
    <div className="scale-sensitivity" role="group" aria-label="Population sensitivity conclusion">
      {sensitivities.map(({ model, value }) => <div className={value !== null && value === highestSensitivity && value > 0 ? "scale-sensitivity__emphasis" : "scale-sensitivity__neutral"} key={model}><span>{model}</span><strong>{value === null ? "n/a" : `${value > 0 ? "+" : ""}${value.toFixed(1)}%`}</strong><small>final success when G doubles</small></div>)}
    </div>
  </div>;
}

export function HomeScalingSummary() {
  const { data, error } = useTaskData("scaling");
  if (error) return <div className="empty-state">Scaling results could not be loaded.</div>;
  if (!data) return <div className="empty-state">Loading scaling results…</div>;
  return <ScalingSummary rows={data.finalResults as unknown as ScaleResult[]}/>;
}
