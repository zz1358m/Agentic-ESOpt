import { useEffect, useRef, useState } from "react";
import { GenerationControl } from "../components/GenerationControl";
import { MetricChart } from "../components/MetricChart";
import { LoadingState, PageFrame } from "../components/SiteChrome";
import { BaselineToggle, DataProvenance, ResultTable, SectionHeading, SegmentedControl, TaskHero } from "../components/TaskPrimitives";
import { TraceViewer } from "../components/TraceViewer";
import { useTaskData } from "../hooks/useTaskData";
import { useUrlNumber } from "../hooks/useUrlNumber";
import { useUrlString } from "../hooks/useUrlString";
import { siteHref } from "../site";
import type { CaseCheckpoint } from "../types";

type DocCase = { id: string; label: string; image: string; question: string; answers: string[]; source: string; checkpoints: CaseCheckpoint[] };

export function DocVqaPage() {
  const { data, error } = useTaskData("docvqa");
  const [generation, setGeneration] = useUrlNumber("generation", 19);
  const [caseId, setCaseId] = useUrlString("case", "docvqa_53536");
  const [turn, setTurn] = useUrlNumber("turn", 1);
  const [split, setSplit] = useUrlString("split", "Periodic Evaluation");
  const [baselines, setBaselines] = useUrlString("baselines", "off");
  const [zoom, setZoom] = useState(1);
  const [dragging, setDragging] = useState(false);
  const viewport = useRef<HTMLDivElement>(null);
  const dragOrigin = useRef({ x: 0, y: 0, left: 0, top: 0 });
  const previousReplay = useRef(`${caseId}:${generation}`);
  useEffect(() => {
    const replay = `${caseId}:${generation}`;
    if (replay !== previousReplay.current) {
      previousReplay.current = replay;
      setTurn(1);
    }
  }, [generation, caseId, setTurn]);
  useEffect(() => setZoom(1), [caseId]);
  if (!data) return <LoadingState error={error} />;
  const cases = data.cases as unknown as DocCase[];
  const selectedCase = cases.find((item) => item.id === caseId) ?? cases[0];
  const generations = selectedCase.checkpoints.map((item) => item.generation);
  const effectiveGeneration = generations.includes(generation) ? generation : generations[0];
  const checkpoint = selectedCase.checkpoints.find((item) => item.generation === effectiveGeneration)!;
  const effectiveSplit = split === "Train" ? "Train" : "Periodic Evaluation";
  const visibleCurves = data.curves.filter((curve) => effectiveSplit === "Train" ? curve.kind === "train" : curve.kind === "eval");
  const startDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const node = viewport.current;
    if (!node) return;
    node.setPointerCapture(event.pointerId);
    dragOrigin.current = { x: event.clientX, y: event.clientY, left: node.scrollLeft, top: node.scrollTop };
    setDragging(true);
  };
  const moveDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const node = viewport.current;
    if (!node || !dragging) return;
    node.scrollLeft = dragOrigin.current.left - (event.clientX - dragOrigin.current.x);
    node.scrollTop = dragOrigin.current.top - (event.clientY - dragOrigin.current.y);
  };
  return (
    <PageFrame>
      <TaskHero kicker="Vision · OCR tool use" title="Document" accent="Visual Question Answering" summary="Inspect the source document and the exact OCR/tool observations that the agent used before returning a short answer." metric="ANLS Mean@4" value="0.5043" detail="Agentic ESOpt + No Skill" />
      <section className="section-shell doc-case-control"><label><span className="control-label">Selected document case</span><select value={selectedCase.id} onChange={(event) => setCaseId(event.target.value)}>{cases.map((item) => <option value={item.id} key={item.id}>{item.label}</option>)}</select></label><span className="data-status"><i />Full replay at Base, 9, 19, 29, and 39</span></section>
      <section className="section-shell doc-workbench">
        <div className="document-pane"><div className="pane-head"><div><span className="eyebrow">Source document</span><h3>{selectedCase.label}</h3></div><div className="zoom-controls"><button type="button" aria-label="Zoom out" onClick={()=>setZoom((v)=>Math.max(.6,v-.2))}>−</button><span>{Math.round(zoom*100)}%</span><button type="button" aria-label="Zoom in" onClick={()=>setZoom((v)=>Math.min(2,v+.2))}>+</button></div></div><div ref={viewport} className={`document-viewport ${dragging ? "dragging" : ""}`} onPointerDown={startDrag} onPointerMove={moveDrag} onPointerUp={() => setDragging(false)} onPointerCancel={() => setDragging(false)}><img draggable={false} src={siteHref(selectedCase.image)} alt={`Scanned source document for ${selectedCase.label}`} style={{width:`${zoom*100}%`}} /></div><p className="fine-print">Zoom with the controls, then drag the document to pan. OCR boxes are omitted because the retained logs do not provide coordinates.</p></div>
        <div className="doc-answer-pane"><span className="eyebrow">Question</span><h2>{selectedCase.question}</h2><div className="answer-comparison"><div><span>Model answer</span><strong>{checkpoint.prediction}</strong></div><div><span>Reference</span><strong>{selectedCase.answers.join(" · ")}</strong></div></div><div className="score-pills"><span>ANLS <b>{checkpoint.anls?.toFixed(2)}</b></span><span>Accuracy <b>{checkpoint.acc?.toFixed(0)}</b></span><span className={checkpoint.score>.5?"success-pill":"failure-pill"}>{checkpoint.score>.5?"Matched":"Missed"}</span></div><GenerationControl generations={generations} value={effectiveGeneration} onChange={setGeneration} /></div>
      </section>
      <section className="section-shell trajectory-section"><SectionHeading eyebrow="OCR replay" title={`From pixels to an answer · ${effectiveGeneration < 0 ? "Base model" : `Generation ${effectiveGeneration}`}`}>This second timeline follows the retained ReAct turns inside the selected checkpoint.</SectionHeading><TraceViewer checkpoint={checkpoint} turn={turn} onTurnChange={setTurn} /></section>
      <section className="section-shell curve-section"><SectionHeading eyebrow="Learning curves" title="Training reward and held-out evaluation" aside={<SegmentedControl label="Curve split" options={["Train","Periodic Evaluation"]} value={effectiveSplit} onChange={setSplit}/>}>This control changes the aggregate curve below; the document and OCR replay remain tied to the selected retained checkpoint.</SectionHeading><MetricChart series={visibleCurves} generation={effectiveGeneration} title={effectiveSplit === "Train" ? "Training reward" : "Held-out ANLS"} valueFormatter={(value)=>value.toFixed(2)} /></section>
      <section className="section-shell results-section"><SectionHeading eyebrow="Final evaluation" title="Answer quality across four samples" aside={<BaselineToggle checked={baselines==="on"} onChange={(value)=>setBaselines(value?"on":"off")} />}/><ResultTable rows={data.finalResults} showBaselines={baselines==="on"} columns={[{key:"method",label:"Method"},{key:"anlsMean",label:"ANLS Mean@4"},{key:"anlsPass",label:"ANLS Pass@4"},{key:"accMean",label:"Accuracy Mean@4",format:(v)=>`${v}%`},{key:"accPass",label:"Accuracy Pass@4",format:(v)=>`${v}%`}]} /><DataProvenance metadata={data.metadata}/></section>
    </PageFrame>
  );
}
