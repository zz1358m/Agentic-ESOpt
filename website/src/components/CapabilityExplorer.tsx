import { useMemo } from "react";

import { useTaskData } from "../hooks/useTaskData";
import { useUrlNumber } from "../hooks/useUrlNumber";
import { useUrlString } from "../hooks/useUrlString";
import type { CurveSeries, TaskPayload } from "../types";
import { siteHref } from "../site";
import { GenerationControl } from "./GenerationControl";
import { HighlightedCode } from "./HighlightedCode";
import { MetricChart } from "./MetricChart";

export type CapabilityTask = "sudoku" | "webarena" | "ahd";

type CapabilityCheckpoint = {
  optimizationStep: number;
  aggregateMetric?: number;
  aggregateMetricLabel?: string;
  score?: number;
  prediction?: number[][];
  feedback?: string;
  output?: string;
  outputUnavailable?: boolean;
  objective?: number;
  algorithm?: string;
  heuristic?: string;
};

type CapabilityCase = {
  id: string;
  goal?: string;
  maskCount?: number;
  puzzle?: number[][];
  solution?: number[][];
  evidenceScope?: string;
  capabilityCheckpoints?: CapabilityCheckpoint[];
};

type TaskDefinition = {
  label: string;
  hint: string;
  path: string;
  initialCheckpoint: "first" | "last";
  sliderLabel: string;
  lowerIsBetter: boolean;
  formatMetric: (value: number) => string;
  formatReading: (value: number) => string;
  formatDelta: (value: number) => string;
  selectCurves: (curves: CurveSeries[]) => CurveSeries[];
};

const TASKS: Record<CapabilityTask, TaskDefinition> = {
  sudoku: { label: "Sudoku", hint: "decoding", path: "tasks/sudoku/", initialCheckpoint: "first", sliderLabel: "ES generation", lowerIsBetter: false, formatMetric: (value) => `${(value * 100).toFixed(0)}%`, formatReading: (value) => `${(value * 100).toFixed(1)}%`, formatDelta: (value) => `${(value * 100).toFixed(1)} points`, selectCurves: (curves) => curves.filter((curve) => curve.id === "mask5-stage3-recheck-eval") },
  webarena: { label: "WebArena", hint: "web outcome", path: "tasks/webarena/", initialCheckpoint: "last", sliderLabel: "Evaluation epoch", lowerIsBetter: false, formatMetric: (value) => `${(value * 100).toFixed(0)}%`, formatReading: (value) => `${(value * 100).toFixed(1)}%`, formatDelta: (value) => `${(value * 100).toFixed(1)} points`, selectCurves: (curves) => curves.filter((curve) => curve.id === "esopt-eval") },
  ahd: { label: "AHD", hint: "heuristic code", path: "tasks/ahd/", initialCheckpoint: "last", sliderLabel: "Search generation", lowerIsBetter: true, formatMetric: (value) => value.toFixed(3), formatReading: (value) => value.toFixed(5), formatDelta: (value) => value.toFixed(5), selectCurves: (curves) => curves.filter((curve) => Boolean(curve.capabilityCurve)) },
};

function phaseLabel(index: number, length: number) {
  if (index === 0) return "Early";
  if (index === length - 1) return "Late";
  return "Middle";
}

function SudokuSnapshot({ puzzle, prediction, previousPrediction, solution }: { puzzle: number[][]; prediction: number[][]; previousPrediction?: number[][]; solution: number[][] }) {
  const columns = puzzle[0]?.length ?? 9;
  return (
    <div className="sudoku-snapshot" role="grid" aria-label="Model Sudoku prediction" style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
      {prediction.map((row, rowIndex) => <div className="sudoku-snapshot__row" role="row" key={rowIndex}>
        {row.map((value, columnIndex) => {
          const given = puzzle[rowIndex]?.[columnIndex] !== 0;
          const state = given ? "given" : value === solution[rowIndex]?.[columnIndex] ? "correct" : "conflict";
          const changed = previousPrediction !== undefined && value !== previousPrediction[rowIndex]?.[columnIndex];
          return <div role="gridcell" aria-label={`Row ${rowIndex + 1} column ${columnIndex + 1}, ${value}, ${state}${changed ? ", changed since previous checkpoint" : ""}`} data-state={state} data-changed={changed || undefined} key={`${rowIndex}-${columnIndex}`}>{value || "·"}</div>;
        })}
      </div>)}
    </div>
  );
}

type DiffLine = { kind: "same" | "added" | "removed"; text: string };

export function diffLines(before: string, after: string): DiffLine[] {
  const left = before.split("\n");
  const right = after.split("\n");
  const lengths = Array.from({ length: left.length + 1 }, () => Array(right.length + 1).fill(0));
  for (let i = left.length - 1; i >= 0; i -= 1) {
    for (let j = right.length - 1; j >= 0; j -= 1) {
      lengths[i][j] = left[i] === right[j] ? lengths[i + 1][j + 1] + 1 : Math.max(lengths[i + 1][j], lengths[i][j + 1]);
    }
  }
  const result: DiffLine[] = [];
  let i = 0;
  let j = 0;
  while (i < left.length || j < right.length) {
    if (i < left.length && j < right.length && left[i] === right[j]) {
      result.push({ kind: "same", text: left[i] }); i += 1; j += 1;
    } else if (j < right.length && (i === left.length || lengths[i][j + 1] > lengths[i + 1][j])) {
      result.push({ kind: "added", text: right[j] }); j += 1;
    } else {
      result.push({ kind: "removed", text: left[i] }); i += 1;
    }
  }
  return result;
}

function sudokuProgress(selectedCase: CapabilityCase, checkpoint: CapabilityCheckpoint, previous?: CapabilityCheckpoint) {
  const masked = (selectedCase.puzzle ?? []).flatMap((row, rowIndex) => row.map((value, columnIndex) => ({ value, rowIndex, columnIndex }))).filter((cell) => cell.value === 0);
  const correct = masked.filter(({ rowIndex, columnIndex }) => checkpoint.prediction?.[rowIndex]?.[columnIndex] === selectedCase.solution?.[rowIndex]?.[columnIndex]).length;
  const previousCorrect = previous && masked.filter(({ rowIndex, columnIndex }) => previous.prediction?.[rowIndex]?.[columnIndex] === selectedCase.solution?.[rowIndex]?.[columnIndex]).length;
  const change = previousCorrect === undefined ? "This is the first retained prediction." : `${correct - previousCorrect >= 0 ? "+" : ""}${correct - previousCorrect} correct masked cells since the previous checkpoint.`;
  return `${checkpoint.score ? "Solved" : "Still failing"}: ${correct}/${masked.length} masked cells match the solution. ${change}`;
}

function AhdCodeDiff({ before, after }: { before?: string; after: string }) {
  if (before === undefined) return null;
  return <details className="capability-diff" open><summary>Code changes from previous checkpoint</summary><pre role="region" aria-label="Code changes from previous checkpoint" tabIndex={0}>{diffLines(before, after).map((line, index) => <span className={`diff-line diff-line--${line.kind}`} key={`${index}-${line.text}`}>{line.kind === "added" ? "+" : line.kind === "removed" ? "−" : " "} {line.text || " "}</span>)}</pre></details>;
}

type ArtifactProps = { selectedCase: CapabilityCase; checkpoint: CapabilityCheckpoint; previous?: CapabilityCheckpoint };

function SudokuArtifact({ selectedCase, checkpoint, previous }: ArtifactProps) {
  return <>
    <div className="capability-artifact__head"><span>Model prediction</span><strong className={checkpoint.score ? "success-text" : "failure-text"}>{checkpoint.score ? "Success" : "Failure"}</strong></div>
    <SudokuSnapshot puzzle={selectedCase.puzzle ?? []} prediction={checkpoint.prediction ?? []} previousPrediction={previous?.prediction} solution={selectedCase.solution ?? []}/>
    <div className="capability-change"><strong>What changed</strong><p>{sudokuProgress(selectedCase, checkpoint, previous)}</p></div>
    <p className="capability-feedback">{checkpoint.feedback}</p>
  </>;
}

function WebArenaArtifact({ selectedCase, checkpoint, previous }: ArtifactProps) {
  const outcome = checkpoint.score ? "Succeeded" : "Still failing";
  const transition = previous ? `The case score ${previous.score === checkpoint.score ? "remains" : "changed"} ${previous.score ?? 0} → ${checkpoint.score ?? 0}.` : "This is the first retained case score.";
  const hasRetainedFinalOutput = Boolean(selectedCase.capabilityCheckpoints?.at(-1)?.output);
  return <>
    <span className="eyebrow">Fixed browser task</span>
    <h3>{selectedCase.goal}</h3>
    <div className="capability-artifact__head"><span>Selected-case result</span><strong className={checkpoint.score ? "success-text" : "failure-text"}>{checkpoint.score ? "Success" : "Failure"}</strong></div>
    {checkpoint.outputUnavailable
      ? <div className="webarena-score-evidence"><span>Score-only checkpoint</span><strong>{checkpoint.score ?? 0} / 1</strong><p>The training log retains this task&apos;s exact outcome at this epoch. {hasRetainedFinalOutput ? "A readable answer is available at the final evaluation." : "No readable answer is attached to this case."}</p></div>
      : <div className="outcome-answer"><span>Retained final output</span><p>{checkpoint.output}</p></div>}
    <div className="capability-change"><strong>What changed</strong><p>{outcome}. {transition} {checkpoint.outputUnavailable ? "This checkpoint is backed by the linked case score." : "The final readable answer is retained."}</p></div>
  </>;
}

function AhdArtifact({ checkpoint, previous }: ArtifactProps) {
  const objectiveDelta = previous && checkpoint.objective !== undefined && previous.objective !== undefined ? checkpoint.objective - previous.objective : undefined;
  return <>
    <div className="capability-artifact__head"><span>Best heuristic at this search generation</span><strong>{checkpoint.objective?.toFixed(5)}</strong></div>
    <p className="capability-feedback">{checkpoint.algorithm}</p>
    <div className="capability-change"><strong>What changed</strong><p>{objectiveDelta === undefined ? "This is the first retained heuristic version." : `${objectiveDelta < 0 ? "Improved" : "Changed"} the objective by ${Math.abs(objectiveDelta).toFixed(5)}; the exact line changes are highlighted below.`}</p></div>
    <AhdCodeDiff before={previous?.heuristic} after={checkpoint.heuristic ?? ""}/>
    <pre className="code-block capability-code" role="region" aria-label="Retained heuristic source code" tabIndex={0}><HighlightedCode>{checkpoint.heuristic ?? ""}</HighlightedCode></pre>
  </>;
}

const TASK_ARTIFACTS = { sudoku: SudokuArtifact, webarena: WebArenaArtifact, ahd: AhdArtifact };

export function CapabilityPanel({ task, data }: { task: CapabilityTask; data: TaskPayload }) {
  const selectedCase = (data.cases as CapabilityCase[]).find((item) => item.capabilityCheckpoints?.length);
  const checkpoints = useMemo(() => selectedCase?.capabilityCheckpoints ?? [], [selectedCase]);
  const definition = TASKS[task];
  const defaultStep = (definition.initialCheckpoint === "last" ? checkpoints.at(-1) : checkpoints[0])?.optimizationStep;
  const [step, setStep] = useUrlNumber(`cap_${task}_step`, defaultStep ?? 0);
  const Artifact = TASK_ARTIFACTS[task];

  if (!selectedCase || !checkpoints.length) return <div className="empty-state">No linked capability checkpoints were retained.</div>;
  const generations = checkpoints.map((item) => item.optimizationStep);
  const effectiveStep = generations.includes(step) ? step : defaultStep ?? generations[0];
  const currentIndex = Math.max(0, generations.indexOf(effectiveStep));
  const current = checkpoints[currentIndex];
  const previous = currentIndex > 0 ? checkpoints[currentIndex - 1] : undefined;
  const curves = definition.selectCurves(data.curves);
  const metric = current.aggregateMetric ?? current.objective;
  const metricLabel = current.aggregateMetricLabel ?? "Best heuristic objective";
  const delta = metric !== undefined && currentIndex > 0
    ? metric - (previous?.aggregateMetric ?? previous?.objective ?? metric)
    : undefined;
  const quickIndexes = [...new Set([0, Math.floor((checkpoints.length - 1) / 2), checkpoints.length - 1])];
  const sliderLabel = definition.sliderLabel;

  return (
    <div className="capability-panel" data-task={task}>
      <div className="capability-panel__summary">
        <div><span className="eyebrow">{definition.label} · same case</span><h3>{selectedCase.id}</h3><p>{selectedCase.evidenceScope}</p></div>
        <div className="capability-reading"><span>{metricLabel}</span><strong>{definition.formatReading(metric ?? 0)}</strong><small>{delta === undefined ? "Initial retained checkpoint" : `${delta >= 0 ? "+" : ""}${definition.formatDelta(delta)} from previous checkpoint`}</small></div>
      </div>
      <div className="capability-quick" aria-label="Capability checkpoint shortcuts">
        {quickIndexes.map((index) => <button type="button" className={index === currentIndex ? "active" : ""} aria-pressed={index === currentIndex} onClick={() => setStep(generations[index])} key={generations[index]}><span>{phaseLabel(index, checkpoints.length)}</span><strong>{generations[index] < 0 ? "Base" : generations[index]}</strong></button>)}
      </div>
      <div className="capability-panel__grid">
        <div>
          <MetricChart series={curves} generation={effectiveStep} title={`${definition.label} capability curve`} valueFormatter={definition.formatMetric} lowerIsBetter={definition.lowerIsBetter} xLabel={sliderLabel}/>
          <GenerationControl generations={generations} value={effectiveStep} onChange={setStep} label={sliderLabel}/>
        </div>
        <article className="capability-artifact">
          <div className="capability-artifact__step"><span>{sliderLabel}</span><strong>{effectiveStep < 0 ? "Base" : effectiveStep}</strong></div>
          <Artifact selectedCase={selectedCase} checkpoint={current} previous={previous}/>
        </article>
      </div>
    </div>
  );
}

export function CapabilityExplorer() {
  const sudoku = useTaskData("sudoku");
  const webarena = useTaskData("webarena");
  const ahd = useTaskData("ahd");
  return <CapabilityExplorerView resources={{ sudoku, webarena, ahd }}/>;
}

type CapabilityResource = { data: TaskPayload | null; error: string };

export function CapabilityExplorerView({ resources }: { resources: Record<CapabilityTask, CapabilityResource> }) {
  const [taskParam, setTask] = useUrlString("cap_task", "sudoku");
  const task = (Object.keys(TASKS) as CapabilityTask[]).includes(taskParam as CapabilityTask) ? taskParam as CapabilityTask : "sudoku";
  const current = resources[task];
  return <div className="capability-explorer">
    <div className="capability-task-tabs" role="group" aria-label="Capability task">
      {(Object.keys(TASKS) as CapabilityTask[]).map((item) => <button type="button" className={item === task ? "active" : ""} aria-pressed={item === task} onClick={() => setTask(item)} key={item}><span>{TASKS[item].label}</span><small>{TASKS[item].hint}</small></button>)}
    </div>
    {current.error ? <div className="empty-state">{TASKS[task].label} capability data could not be loaded.</div> : current.data ? <CapabilityPanel task={task} data={current.data} key={task}/> : <div className="empty-state">Loading linked checkpoints…</div>}
    <div className="capability-explorer__footer"><a className="text-link" href={siteHref(TASKS[task].path)}>Open full {TASKS[task].label} explorer →</a></div>
  </div>;
}
