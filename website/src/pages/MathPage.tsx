import { useEffect, useRef } from "react";
import { GenerationControl } from "../components/GenerationControl";
import { MetricChart } from "../components/MetricChart";
import { LoadingState, PageFrame } from "../components/SiteChrome";
import { BaselineToggle, DataProvenance, ResultTable, SectionHeading, SegmentedControl, TaskHero } from "../components/TaskPrimitives";
import { TraceViewer } from "../components/TraceViewer";
import { useTaskData } from "../hooks/useTaskData";
import { useUrlNumber } from "../hooks/useUrlNumber";
import { useUrlString } from "../hooks/useUrlString";
import type { CaseCheckpoint, TaskPayload } from "../types";

type MathCase = { id: string; label: string; dataset: string; question: string; answer: string; source: string; checkpoints: CaseCheckpoint[] };

export function mathGenerationOptions(data: TaskPayload, split: "Train" | "Periodic Evaluation" = "Train", dataset = "DAPO"): number[] {
  const curve = split === "Train"
    ? data.curves.find((item) => item.kind === "train")
    : data.curves.find((item) => item.id === (dataset === "DAPO" ? "dapo_eval" : "aime_eval"));
  return curve?.points.map((point) => point.generation) ?? [];
}

export function MathPage() {
  const { data, error } = useTaskData("math");
  const [generation, setGeneration] = useUrlNumber("generation", 19);
  const [dataset, setDataset] = useUrlString("dataset", "DAPO");
  const [caseId, setCaseId] = useUrlString("case", "dapo_5585520b-e89c-4c8c-afdb-172b0137f0e2");
  const [turn, setTurn] = useUrlNumber("turn", 1);
  const [split, setSplit] = useUrlString("split", "Periodic Evaluation");
  const [baselines, setBaselines] = useUrlString("baselines", "off");
  const previousReplay = useRef(`${caseId}:${generation}`);
  useEffect(() => {
    const replay = `${caseId}:${generation}`;
    if (replay !== previousReplay.current) {
      previousReplay.current = replay;
      setTurn(1);
    }
  }, [caseId, generation, setTurn]);
  if (!data) return <LoadingState error={error} />;
  const cases = data.cases as unknown as MathCase[];
  const availableDatasets = [...new Set(cases.map((item) => item.dataset))];
  const effectiveDataset = availableDatasets.includes(dataset) ? dataset : availableDatasets[0];
  const datasetCases = cases.filter((item) => item.dataset === effectiveDataset);
  const selectedCase = datasetCases.find((item) => item.id === caseId) ?? datasetCases[0];
  const scored = selectedCase.checkpoints.map((item) => item.generation);
  const effectiveSplit = split === "Train" ? "Train" : "Periodic Evaluation";
  const generations = mathGenerationOptions(data, effectiveSplit, effectiveDataset);
  const effectiveGeneration = generations.includes(generation) ? generation : generations.at(-1) ?? 0;
  const checkpoint = selectedCase.checkpoints.find((item) => item.generation === effectiveGeneration);
  const visibleCurves = effectiveSplit === "Train"
    ? data.curves.filter((curve) => curve.kind === "train")
    : data.curves.filter((curve) => curve.id === (effectiveDataset === "DAPO" ? "dapo_eval" : "aime_eval"));
  const displayProblem = selectedCase.question
    .replace(/^Solve the following math problem step by step\.[\s\S]*?\n\n/, "")
    .replace(/\n\nRemember[\s\S]*$/, "")
    .replaceAll("$", "")
    .replaceAll("\\equiv", "≡")
    .replace(/\\pmod\{(\d+)\}/g, "(mod $1)");
  return (
    <PageFrame>
      <TaskHero kicker="Reasoning · ReAct tools" title="Tool-Using" accent="Math Reasoning" summary="Follow one held-out problem as the model learns to reason, call a command-line tool, and submit an exact final answer." metric="AIME 2026 Mean@4" value="70.8%" detail="Agentic ESOpt + No Skill" />
      <section className="section-shell explorer-section">
        <div className="explorer-toolbar"><label><span className="control-label">Dataset</span><select value={effectiveDataset} onChange={(event) => { const nextDataset = event.target.value; const nextCase = cases.find((item) => item.dataset === nextDataset)!; setDataset(nextDataset); setCaseId(nextCase.id); }}>{availableDatasets.map((item) => <option key={item}>{item}</option>)}</select></label><label><span className="control-label">Selected case</span><select value={selectedCase.id} onChange={(event) => setCaseId(event.target.value)}>{datasetCases.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label><SegmentedControl label="Curve split" options={["Train","Periodic Evaluation"]} value={effectiveSplit} onChange={(value)=>{setSplit(value);const options=mathGenerationOptions(data,value,effectiveDataset);setGeneration(options.includes(effectiveGeneration)?effectiveGeneration:options.at(-1)??0);}}/><div><span className="control-label">Skill setting</span><strong>Agentic ESOpt · No Skill</strong></div><div className="availability"><span>Full trajectory</span><div>{scored.map((item) => <i key={item} className={item === effectiveGeneration ? "active" : ""}>{item}</i>)}</div></div></div>
        <div className="explorer-grid explorer-grid--math">
          <MetricChart series={visibleCurves} generation={effectiveGeneration} title={effectiveSplit === "Train" ? "Training reward" : `Periodic ${effectiveDataset} accuracy`} valueFormatter={(value) => `${(value * 100).toFixed(0)}%`} />
          <article className="case-card"><span className="eyebrow">Selected task · {effectiveDataset}</span><h3>{selectedCase.label}</h3><p>{displayProblem}</p><div className="reference-answer"><span>Reference answer</span><strong>{selectedCase.answer}</strong></div></article>
        </div>
        <GenerationControl generations={generations} value={effectiveGeneration} onChange={setGeneration} />
        <div className="trajectory-section">
          <SectionHeading eyebrow="Capability replay" title={checkpoint ? `What the agent did at generation ${effectiveGeneration}` : `Generation ${effectiveGeneration} metric only`}>{checkpoint ? "The ReAct trace below is copied from the recorded checkpoint. Use the second timeline to inspect tool calls and observations." : "This generation has an aggregate training point, but no retained qualitative trajectory."}</SectionHeading>
          {checkpoint ? <TraceViewer checkpoint={checkpoint} turn={turn} onTurnChange={setTurn} /> : <div className="missing-trajectory"><span>○</span><strong>Trajectory not recorded</strong><p>Move to generation 9, 19, 24, or 25 for a complete replay.</p></div>}
        </div>
      </section>
      <section className="section-shell results-section"><SectionHeading eyebrow="Final evaluation" title="Performance across two math distributions" aside={<BaselineToggle checked={baselines==="on"} onChange={(value)=>setBaselines(value?"on":"off")} />}>Mean@4 averages four answers; Pass@4 records whether any of the four succeeds.</SectionHeading><ResultTable rows={data.finalResults} showBaselines={baselines==="on"} columns={[{key:"method",label:"Method"},{key:"dapoMean",label:"DAPO Mean@4",format:(v)=>`${v}%`},{key:"dapoPass",label:"DAPO Pass@4",format:(v)=>`${v}%`},{key:"aimeMean",label:"AIME Mean@4",format:(v)=>`${v}%`},{key:"aimePass",label:"AIME Pass@4",format:(v)=>`${v}%`}]} /><DataProvenance metadata={data.metadata} /></section>
    </PageFrame>
  );
}
