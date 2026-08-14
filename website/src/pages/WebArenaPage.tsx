import { GenerationControl } from "../components/GenerationControl";
import { CapabilityPanel } from "../components/CapabilityExplorer";
import { MetricChart } from "../components/MetricChart";
import { LoadingState, PageFrame } from "../components/SiteChrome";
import { BaselineToggle, DataProvenance, ResultTable, SectionHeading, SegmentedControl, TaskHero } from "../components/TaskPrimitives";
import { useTaskData } from "../hooks/useTaskData";
import { useUrlNumber } from "../hooks/useUrlNumber";
import { useUrlString } from "../hooks/useUrlString";

type Outcome = { setting:string; run:number; goal:string; site:string; hard:number; soft:number; turns:number; answer:string; failureReason:string };
type WebCase = { id:string; taskId:number; site:string; goal:string; outcomes:Outcome[] };
const settings = ["No Skill","Agentic ESOpt","Trace2Skill","Agentic ESOpt + Trace2Skill"];

export function WebArenaPage() {
  const { data, error } = useTaskData("webarena");
  const [epoch, setEpoch] = useUrlNumber("epoch",70);
  const [site,setSite]=useUrlString("site","shopping");
  const [caseId,setCaseId]=useUrlString("case","task-4");
  const [setting,setSetting]=useUrlString("setting","Agentic ESOpt");
  const [run,setRun]=useUrlNumber("run",1);
  const [showBaselines,setShowBaselines]=useUrlString("baselines","off");
  if(!data) return <LoadingState error={error}/>;
  const cases=data.cases as unknown as WebCase[];
  const sites=[...new Set(cases.map((item)=>item.site))];
  const effectiveSite=sites.includes(site)?site:sites[0];
  const siteCases=cases.filter((item)=>item.site===effectiveSite);
  const selectedCase=siteCases.find((item)=>item.id===caseId) ?? siteCases[0];
  const effectiveSetting=settings.includes(setting)?setting:settings[0];
  const effectiveRun=[1,2,3].includes(run)?run:1;
  const outcome=selectedCase.outcomes.find((item)=>item.setting===effectiveSetting&&item.run===effectiveRun) ?? selectedCase.outcomes[0];
  const comparison=settings.map((item)=>selectedCase.outcomes.find((candidate)=>candidate.setting===item&&candidate.run===effectiveRun)).filter((item):item is Outcome=>Boolean(item));
  const epochs=data.curves[0].points.map((point)=>point.generation);
  const effectiveEpoch=epochs.includes(epoch)?epoch:epochs.at(-1)!;
  return <PageFrame>
    <TaskHero kicker="Goal-conditioned web control" title="WebArena-Lite" accent="Outcome Explorer" summary="Compare what four model-and-skill settings ultimately accomplished on the same browser task—using only the outcomes retained in the original logs." metric="Dataset success" value="36.16%" detail="Agentic ESOpt + No Skill" />
    <section className="section-shell capability-section capability-section--task"><SectionHeading eyebrow="Same-task evolution" title="The curve and task outcome move together.">Task 4 has the maximum three-repeat No Skill-to-ESOpt gain: 0% to 100%. It is shown at epochs 10, 50, and 70; a readable answer is retained for the final checkpoint.</SectionHeading><CapabilityPanel task="webarena" data={data}/></section>
    <section className="section-shell explorer-section"><div className="outcome-controls outcome-controls--primary"><label><span>Site / category</span><select value={effectiveSite} onChange={(event)=>{const nextSite=event.target.value;const nextCase=cases.find((item)=>item.site===nextSite)!;setSite(nextSite);setCaseId(nextCase.id);}}>{sites.map((item)=><option key={item}>{item}</option>)}</select></label><label><span>Task</span><select value={selectedCase.id} onChange={(event)=>setCaseId(event.target.value)}>{siteCases.map((item)=><option value={item.id} key={item.id}>#{item.taskId} · {item.goal}</option>)}</select></label><label><span>Setting</span><select value={effectiveSetting} onChange={(event)=>setSetting(event.target.value)}>{settings.map((item)=><option key={item}>{item}</option>)}</select></label><SegmentedControl label="Evaluation repeat" options={[1,2,3]} value={effectiveRun} onChange={setRun} format={(value)=>`Run ${value}`}/></div><div className="explorer-grid explorer-grid--webarena"><div><MetricChart series={data.curves} generation={effectiveEpoch} title="Periodic WebArena-Lite evaluation" valueFormatter={(value)=>`${Math.round(value*100)}%`}/><GenerationControl generations={epochs} value={effectiveEpoch} onChange={setEpoch} label="Evaluation epoch"/></div><article className="outcome-card"><div className="browser-chrome"><i/><i/><i/><span>{selectedCase.site} · task {selectedCase.taskId}</span></div><span className="eyebrow">Natural-language goal</span><h3>{outcome.goal}</h3><div className="outcome-answer"><span>Final answer</span><p>{outcome.answer || "No final answer was recorded."}</p></div>{outcome.failureReason&&<div className="failure-reason"><span>Recorded failure reason</span><p>{outcome.failureReason}</p></div>}<div className="outcome-metrics"><div><span>Status</span><strong className={outcome.hard?"success-text":"failure-text"}>{outcome.hard?"Success":"Failure"}</strong></div><div><span>Turns</span><strong>{outcome.turns}</strong></div><div><span>Hard / soft</span><strong>{outcome.hard.toFixed(0)} / {outcome.soft.toFixed(2)}</strong></div></div></article></div><div className="outcome-comparison" aria-label="Same-task setting comparison">{comparison.map((item)=><article key={item.setting} className={item.setting===effectiveSetting?"active":""}><span>{item.setting}</span><strong className={item.hard?"success-text":"failure-text"}>{item.hard?"Success":"Failure"}</strong><small>soft {item.soft.toFixed(2)} · {item.turns} turns</small></article>)}</div><aside className="retention-note"><span>ℹ</span><p><strong>Outcome view, not a browser replay.</strong> Per-turn browser observations and actions were not retained in the original logs.</p></aside></section>
    <section className="section-shell results-section"><SectionHeading eyebrow="Final evaluation" title="Success varies across website categories" aside={<BaselineToggle checked={showBaselines==="on"} onChange={(value)=>setShowBaselines(value?"on":"off")}/>}/><ResultTable rows={data.finalResults} showBaselines={showBaselines==="on"} columns={[{key:"method",label:"Method"},{key:"reddit",label:"Reddit",format:(v)=>`${v}%`},{key:"gitlab",label:"GitLab",format:(v)=>`${v}%`},{key:"cms",label:"CMS",format:(v)=>`${v}%`},{key:"map",label:"Map",format:(v)=>`${v}%`},{key:"oss",label:"OSS",format:(v)=>`${v}%`},{key:"average",label:"Average",format:(v)=>`${v}%`}]} /><DataProvenance metadata={data.metadata}/></section>
  </PageFrame>;
}
