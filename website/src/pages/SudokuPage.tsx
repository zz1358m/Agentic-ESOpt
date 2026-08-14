import { GenerationControl } from "../components/GenerationControl";
import { CapabilityPanel } from "../components/CapabilityExplorer";
import { MetricChart } from "../components/MetricChart";
import { LoadingState, PageFrame } from "../components/SiteChrome";
import { BaselineToggle, DataProvenance, ResultTable, SectionHeading, SegmentedControl, TaskHero } from "../components/TaskPrimitives";
import { SudokuBoard } from "../components/SudokuBoard";
import { useTaskData } from "../hooks/useTaskData";
import { useUrlNumber } from "../hooks/useUrlNumber";
import { useUrlString } from "../hooks/useUrlString";

type SudokuCase = { id: string; maskCount: number; puzzle: number[][]; solution: number[][] };

export function SudokuPage() {
  const { data, error } = useTaskData("sudoku");
  const [mask, setMask] = useUrlNumber("mask", 5);
  const [generation, setGeneration] = useUrlNumber("generation", 99);
  const [split, setSplit] = useUrlString("split", "Periodic Evaluation");
  const [method, setMethod] = useUrlString("method", "Agentic ESOpt");
  const [curveBaselines, setCurveBaselines] = useUrlString("curveBaselines", "off");
  const [resultBaselines, setResultBaselines] = useUrlString("resultBaselines", "off");
  if (!data) return <LoadingState error={error}/>;
  const effectiveMask = [5, 10, 15].includes(mask) ? mask : 15;
  const curves = data.curves.filter((curve) => curve.mask === effectiveMask && !curve.configId?.includes("stage3-recheck"));
  const selectedCase = data.cases.find((item) => Number(item.maskCount) === effectiveMask) as unknown as SudokuCase;
  const config = data.configurations.find((item) => Number(item.maskCount) === effectiveMask) as { methods: string[] } | undefined;
  const methods = config?.methods ?? ["Agentic ESOpt"];
  const effectiveMethod = methods.includes(method) ? method : methods[0];
  const effectiveSplit = split === "Train" ? "Train" : "Periodic Evaluation";
  const splitCurves = effectiveSplit === "Train"
    ? curves.filter((curve) => curve.kind === "train")
    : curves.filter((curve) => curve.kind === "eval" && !curve.id.endsWith("train-eval"));
  const primarySeries = splitCurves.filter((curve) => curve.method === effectiveMethod);
  const selectedSeries = curveBaselines === "on" ? [...primarySeries, ...splitCurves.filter((curve) => curve.method !== effectiveMethod)] : primarySeries;
  const generations = primarySeries[0]?.points.map((point) => point.generation) ?? [];
  const nearest = primarySeries[0]?.points.reduce((best, point) => Math.abs(point.generation-generation)<Math.abs(best.generation-generation)?point:best,primarySeries[0].points[0]);
  const initial = primarySeries[0]?.points[0];
  const change = nearest && initial ? nearest.value - initial.value : undefined;
  const agenticFinal = data.finalResults.find((row) => row.method === "Agentic ESOpt");
  const averageFinalSuccess = Number(agenticFinal?.[String(effectiveMask)]);
  return (
    <PageFrame>
      <TaskHero kicker="Long-horizon control" title="Sudoku," accent="One Move at a Time" summary="The model receives terminal reward only after filling every masked cell. Increase the horizon, scrub through training, then try the task yourself." metric="Average final-test success rate" value={Number.isFinite(averageFinalSuccess) ? `${averageFinalSuccess.toFixed(2)}%` : "—"} detail={`Agentic ESOpt · mask ${effectiveMask} · task-level average`} />
      <section className="section-shell capability-section capability-section--task"><SectionHeading eyebrow="Model capability replay" title="One favorable puzzle, five real checkpoints.">The linked mask-5 case has the maximum three-repeat Base-to-final gain without regression among eligible Stage 3 replays. Paper curves for masks 5, 10, and 15 remain available below.</SectionHeading><CapabilityPanel task="sudoku" data={data}/></section>
      <section className="section-shell explorer-section">
        <div className="task-controls"><SegmentedControl label="Masked cells / turn horizon" options={[5,10,15]} value={effectiveMask} onChange={(value)=>{const next=data.configurations.find((item)=>Number(item.maskCount)===value) as {methods:string[]};setMask(value);setMethod(next.methods[0]);setGeneration(effectiveSplit==="Train"?0:99);}} format={(value)=>`${value} turns`} /><SegmentedControl label="Curve split" options={["Train","Periodic Evaluation"]} value={effectiveSplit} onChange={(value)=>{setSplit(value);setGeneration(value==="Train"?(effectiveMethod.startsWith("GRPO")?1:0):99);}}/><label className="fixed-config"><span>Method</span><select value={effectiveMethod} onChange={(event)=>{setMethod(event.target.value);setGeneration(effectiveSplit==="Train"?(event.target.value.startsWith("GRPO")?1:0):99);}}>{methods.map((item)=><option key={item}>{item}</option>)}</select><small>PPO has final results but no retained curve log.</small></label><BaselineToggle checked={curveBaselines==="on"} onChange={(value)=>setCurveBaselines(value?"on":"off")}/><div className="current-reading"><span>Selected {effectiveSplit.toLowerCase()}</span><strong>{nearest ? `${(nearest.value*100).toFixed(1)}%` : "—"}</strong><small>{change === undefined ? "—" : `${change >= 0 ? "+" : ""}${(change*100).toFixed(1)} points from initial`}{nearest?.averageTurns ? ` · ${nearest.averageTurns.toFixed(1)} avg turns` : ""}</small></div></div>
        <div className="explorer-grid explorer-grid--sudoku"><div><MetricChart series={selectedSeries} generation={generation} title={`Mask ${effectiveMask} · ${effectiveSplit}`} valueFormatter={(value)=>`${Math.round(value*100)}%`} /><GenerationControl generations={generations} value={generations.includes(generation)?generation:generations.at(-1) ?? 0} onChange={setGeneration}/></div><article className="board-card"><div className="pane-head"><div><span className="eyebrow">Try the environment</span><h3>{effectiveMask}-cell puzzle</h3></div><span className="demo-badge">Task demo</span></div><SudokuBoard puzzle={selectedCase.puzzle} solution={selectedCase.solution} resetKey={selectedCase.id}/><p className="fine-print">The interactive board demonstrates the task; checkpoint curves report aggregate model performance.</p></article></div>
      </section>
      <section className="section-shell results-section"><SectionHeading eyebrow="Final test" title="Longer horizons change the ordering" aside={<BaselineToggle checked={resultBaselines==="on"} onChange={(value)=>setResultBaselines(value?"on":"off")}/>}>Agentic ESOpt becomes strongest when the delayed trajectory-level credit problem is hardest.</SectionHeading><ResultTable rows={data.finalResults} showBaselines={resultBaselines==="on"} columns={[{key:"method",label:"Method"},{key:"5",label:"5 turns",format:(v)=>`${v}%`},{key:"10",label:"10 turns",format:(v)=>`${v}%`},{key:"15",label:"15 turns",format:(v)=>`${v}%`}]} /><DataProvenance metadata={data.metadata}/></section>
    </PageFrame>
  );
}
