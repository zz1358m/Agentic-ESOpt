import { LoadingState, PageFrame } from "../components/SiteChrome";
import { DataProvenance, SectionHeading, SegmentedControl } from "../components/TaskPrimitives";
import { useTaskData } from "../hooks/useTaskData";
import { useUrlNumber } from "../hooks/useUrlNumber";
import { useUrlString } from "../hooks/useUrlString";
import { buildScalingMatrix, ES_POPULATION_SIZES, MODEL_SIZES, scalingKey, type ScaleResult } from "../scaling";
import type { TaskPayload } from "../types";

export type { ScaleResult } from "../scaling";

type Metric = "best" | "final";

const signed = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;

export function ScalingExplorer({ rows, metadata }: { rows: ScaleResult[]; metadata?: TaskPayload["metadata"] }) {
  const [metricValue, setMetric] = useUrlString("metric", "best");
  const [model, setModel] = useUrlString("model", "9B");
  const [population, setPopulation] = useUrlNumber("population", 16);
  const metric: Metric = metricValue === "final" ? "final" : "best";
  const matrix = buildScalingMatrix(rows);

  if (!matrix) return <section className="section-shell scaling-data-error"><h1 className="display-title"><em>Scaling</em> across model size and ES population</h1><div className="empty-state">The complete 4B/9B × G=8/16 scaling matrix is unavailable.</div></section>;
  const selected = matrix[scalingKey(model === "4B" ? "4B" : "9B", population === 8 ? 8 : 16)];
  const modelComparator = matrix[scalingKey(selected.model === "4B" ? "9B" : "4B", selected.population)];
  const populationComparator = matrix[scalingKey(selected.model, selected.population === 8 ? 16 : 8)];
  const modelAt16 = matrix["9B-16"][metric] - matrix["4B-16"][metric];
  const populationFor4B = matrix["4B-16"][metric] - matrix["4B-8"][metric];
  const populationFor9B = matrix["9B-16"][metric] - matrix["9B-8"][metric];

  const select = (row: ScaleResult) => {
    setModel(row.model);
    setPopulation(row.population);
  };

  return <>
    <section className="scaling-hero section-shell">
      <span className="eyebrow">Sudoku Mask-15 · Vanilla ES ablation · 2 × 2 observed settings</span>
      <h1 className="display-title"><span className="scaling-hero__lead"><em>Scaling</em> across</span>{" "}<span className="scaling-hero__axes">model size and ES population</span></h1>
      <p className="scaling-hero__description">Compare Qwen3.5-4B and 9B at G=8 and G=16; G is the number of sampled perturbation directions per ES update, not physical compute nodes.</p>
    </section>
    <section className="section-shell scaling-explorer">
      <div className="scaling-controls">
        <SegmentedControl label="Displayed result" options={["best", "final"]} value={metric} onChange={setMetric} format={(value) => value === "best" ? "Best checkpoint" : "Final checkpoint"}/>
        <div className="selected-scale">
          <span>Selected configuration</span>
          <strong>{selected.model} · G={selected.population}</strong>
          <b>{selected[metric].toFixed(2)}%</b>
          {modelComparator && <small>{signed(selected[metric] - modelComparator[metric])} points vs {modelComparator.model} at fixed G={selected.population}</small>}
          {populationComparator && <small>{signed(selected[metric] - populationComparator[metric])} points vs G={populationComparator.population} at fixed {selected.model}</small>}
        </div>
      </div>
      <div className="matrix-shell scaling-matrix" aria-label={`${metric === "best" ? "Best checkpoint" : "Final checkpoint"} success by model size and ES population size`}>
        <div/>
        {ES_POPULATION_SIZES.map((value) => <div className="matrix-head" key={value}>G={value}</div>)}
        {MODEL_SIZES.map((modelName) => <div className="matrix-row" key={modelName}>
          <div className="matrix-head">{modelName}</div>
          {ES_POPULATION_SIZES.map((populationSize) => {
            const row = matrix[scalingKey(modelName, populationSize)];
            const active = row === selected;
            return <button type="button" aria-label={`${row.model} model, ES population ${row.population}. Best checkpoint success ${row.best.toFixed(2)}%. Final checkpoint success ${row.final.toFixed(2)}%.`} aria-pressed={active} className={`matrix-cell ${active ? "active" : ""}`} onClick={() => select(row)} key={`${row.model}-${row.population}`}>
              <span>{row.model} model · G={row.population}</span>
              <strong>{row[metric].toFixed(2)}%</strong>
              <div><i style={{ width: `${Math.max(4, row[metric] / 0.4)}%` }}/></div>
              <small>{metric === "best" ? `Final ${row.final.toFixed(2)}%` : `Best ${row.best.toFixed(2)}%`}</small>
            </button>;
          })}
        </div>)}
        <span className="matrix-axis matrix-axis--top">ES population size</span>
        <span className="matrix-axis matrix-axis--side">Model size</span>
      </div>
    </section>
    <section className="section-shell scaling-interpretation">
      <SectionHeading eyebrow="Reading both axes" title="Model size and ES population interact in this recorded ablation."/>
      <div className="interpretation-grid interpretation-grid--scaling">
        <div><strong>{signed(modelAt16)} pts</strong><span>9B vs 4B · fixed G=16 · {metric}</span></div>
        <div><strong>{signed(populationFor4B)} pts</strong><span>G=16 vs G=8 · fixed 4B · {metric}</span></div>
        <div><strong>{signed(populationFor9B)} pts</strong><span>G=16 vs G=8 · fixed 9B · {metric}</span></div>
        <p>The 4B backbone is much more sensitive to ES population size across these four observed settings, each evaluated with three repeats. This two-backbone, two-population ablation is preliminary evidence, not a universal scaling law.</p>
      </div>
      {metadata ? <DataProvenance metadata={{
        ...metadata,
        note: "Model-size scaling compares 4B/9B while holding G fixed. ES population scaling compares G=8/16 while holding the backbone fixed. G is the number of perturbation directions per ES update, not a physical compute-node count.",
      }}/> : null}
    </section>
  </>;
}

export function ScalingPage() {
  const { data, error } = useTaskData("scaling");
  if (!data) return <LoadingState error={error}/>;
  return <PageFrame><ScalingExplorer rows={data.finalResults as unknown as ScaleResult[]} metadata={data.metadata}/></PageFrame>;
}
