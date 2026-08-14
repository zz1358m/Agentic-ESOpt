import { GenerationControl } from "../components/GenerationControl";
import { CapabilityPanel } from "../components/CapabilityExplorer";
import { HighlightedCode } from "../components/HighlightedCode";
import { MetricChart } from "../components/MetricChart";
import { LoadingState, PageFrame } from "../components/SiteChrome";
import { BaselineToggle, DataProvenance, ResultTable, SectionHeading, TaskHero } from "../components/TaskPrimitives";
import { useTaskData } from "../hooks/useTaskData";
import { useAhdSelection, type AhdConfigSelection } from "../hooks/useAhdSelection";
import { useUrlString } from "../hooks/useUrlString";
import type { CSSProperties } from "react";

type AhdConfig={id:string;sourceFiles:string[]}&AhdConfigSelection;
type AhdCheckpoint={generation:number;best:number;bestSoFar:number;candidates:number[];invalidCandidates:number;operator:string};
type AhdReplayCheckpoint={optimizationStep:number;objective:number;algorithm:string;heuristic:string};
type AhdCase={id:string;configId:string;finalHeuristic:string;checkpoints:AhdCheckpoint[];capabilityCheckpoints?:AhdReplayCheckpoint[]};

function buildCandidateHistogram(values:number[],binCount=10){
  const minimum=Math.min(...values);
  const maximum=Math.max(...values);
  if(minimum===maximum)return [{minimum,maximum,count:values.length}];
  const width=(maximum-minimum)/binCount;
  const bins=Array.from({length:binCount},(_,index)=>({minimum:minimum+index*width,maximum:minimum+(index+1)*width,count:0}));
  for(const value of values){
    const index=Math.min(binCount-1,Math.floor((value-minimum)/width));
    bins[index].count+=1;
  }
  return bins;
}

function CandidateHistogram({values,generation}:{values:number[];generation:number}){
  if(!values.length)return <div className="candidate-histogram candidate-histogram--empty" role="img" aria-label={`Candidate objective histogram for generation ${generation}: no valid heuristic programs.`}>No valid objective values were retained for this generation.</div>;
  const bins=buildCandidateHistogram(values);
  const largestBin=Math.max(...bins.map((bin)=>bin.count),1);
  const accessibleBins=bins.map((bin)=>`${bin.minimum.toFixed(2)} to ${bin.maximum.toFixed(2)}: ${bin.count}`).join("; ");
  return <div className="candidate-histogram" role="img" aria-label={`Candidate objective histogram for generation ${generation}: ${values.length} valid candidate evaluations. Lower internal objective is better. Bin counts from lower to higher objective: ${accessibleBins}.`}>
    <div className="candidate-histogram__y-label">Candidate count</div>
    <div className="candidate-histogram__plot">{bins.map((bin,index)=>{const barHeight=bin.count/largestBin*80;return <div className="candidate-histogram__bin" data-count={bin.count} key={index} title={`${bin.minimum.toFixed(2)}–${bin.maximum.toFixed(2)}: ${bin.count} candidate${bin.count===1?"":"s"}`} style={{"--bar-height":`${barHeight}%`} as CSSProperties}><span>{bin.count||""}</span><i className="candidate-histogram__bar" style={{height:`${barHeight}%`}}/></div>;})}</div>
    <div className="candidate-histogram__range"><span>{bins[0].minimum.toFixed(2)}</span><span>{bins.at(-1)!.maximum.toFixed(2)}</span></div>
    <div className="candidate-histogram__axis"><span>Lower internal objective · better</span><span>Higher internal objective · worse</span></div>
  </div>;
}

export function AhdPage(){
  const {data,error}=useTaskData("ahd");
  const {selection,generation,setGeneration,apply}=useAhdSelection();
  const [baselines,setBaselines]=useUrlString("baselines","off");
  if(!data)return <LoadingState error={error}/>;
  const configurations=data.configurations as unknown as AhdConfig[];
  const preferred=configurations.find((item)=>Object.entries(selection).every(([key,value])=>item[key as keyof AhdConfig]===value));
  const selectedConfig=preferred ?? configurations.find((item)=>item.id==="tsp-aco-sample-agentic-1000-r1") ?? configurations[0];
  const selectedCase=(data.cases as unknown as AhdCase[]).find((item)=>item.configId===selectedConfig.id)!;
  const checkpoints=selectedCase.checkpoints;
  const replayCheckpoints=selectedCase.capabilityCheckpoints ?? [];
  const usesReplay=!checkpoints.length&&replayCheckpoints.length>0;
  const selectedCurve=data.curves.filter((curve)=>curve.configId===selectedConfig.id&&Boolean(curve.capabilityCurve)===usesReplay);
  const generations=usesReplay?replayCheckpoints.map((item)=>item.optimizationStep):checkpoints.map((item)=>item.generation);
  const selectedGeneration=generations.includes(generation)?generation:generations.at(-1);
  const current=checkpoints.find((item)=>item.generation===selectedGeneration);
  const replayCurrent=replayCheckpoints.find((item)=>item.optimizationStep===selectedGeneration);
  const finite=current?.candidates.filter(Number.isFinite) ?? [];
  const finalGeneration=generations.at(-1);
  const displayedHeuristic=replayCurrent?.heuristic ?? (selectedGeneration===finalGeneration||!generations.length?selectedCase.finalHeuristic:"");
  const problems=[...new Set(configurations.map((item)=>item.problem))];
  const modes=[...new Set(configurations.filter((item)=>item.problem===selectedConfig.problem).map((item)=>item.mode))];
  const choose=(updates:Partial<AhdConfig>)=>{
    const candidate=configurations.find((item)=>Object.entries({...selectedConfig,...updates}).every(([key,value])=>key==="id"||item[key as keyof AhdConfig]===value))
      ?? configurations.find((item)=>Object.entries(updates).every(([key,value])=>item[key as keyof AhdConfig]===value));
    if(!candidate)return;
    const nextCase=(data.cases as unknown as AhdCase[]).find((item)=>item.configId===candidate.id);
    apply(candidate,nextCase?.checkpoints.at(-1)?.generation ?? 0);
  };
  return <PageFrame>
    <TaskHero kicker="Test-time adaptation · Code design" title="Heuristic" accent="Evolution Explorer" summary="Inspect the retained search evidence across constructive and ACO tasks. Controls expose only configurations backed by original result artifacts." metric="Stage 3 objective" value="5.9026" detail="ACO-TSP · Sample + Agentic ESOpt · generation 50" />
    <section className="section-shell capability-section capability-section--task"><SectionHeading eyebrow="Heuristic capability replay" title="Search changes the program, not model weights.">This is the most favorable eligible replay with multiple retained heuristic versions: objective 6.48937 to 5.90256. It comes from the ACO-TSP Stage 3 execution-side PASS record, which is awaiting designated final review.</SectionHeading><CapabilityPanel task="ahd" data={data}/></section>
    <section className="section-shell explorer-section">
      <div className="ahd-controls">
        <label><span>Problem</span><select value={selectedConfig.problem} onChange={(event)=>choose({problem:event.target.value})}>{problems.map((item)=><option key={item}>{item}</option>)}</select></label>
        <label><span>Mode</span><select value={selectedConfig.mode} onChange={(event)=>choose({mode:event.target.value})}>{modes.map((item)=><option key={item}>{item}</option>)}</select></label>
        <label><span>Outer method</span><select value={selectedConfig.outerMethod} onChange={(event)=>choose({outerMethod:event.target.value})}>{["Sample","EoH"].map((item)=><option key={item}>{item}</option>)}</select></label>
        <label><span>Agentic ESOpt</span><select value={selectedConfig.agenticESOpt?"on":"off"} onChange={(event)=>choose({agenticESOpt:event.target.value==="on"})}><option value="off">Off</option><option value="on">On</option></select></label>
        <label><span>Budget</span><select value={selectedConfig.budget} onChange={(event)=>choose({budget:Number(event.target.value)})}><option value={1000}>1,000</option><option value={2000}>2,000</option></select></label>
        <label><span>Repeat</span><select value={selectedConfig.repeat} onChange={(event)=>choose({repeat:Number(event.target.value)})}>{[1,2,3].map((item)=><option key={item}>{item}</option>)}</select></label>
      </div>
      <div className="explorer-grid explorer-grid--ahd">
        <div><MetricChart series={selectedCurve} generation={selectedGeneration??0} title={usesReplay?"Favorable replay objective":"Best-so-far training objective"} valueFormatter={(value)=>value.toFixed(2)} lowerIsBetter/>{generations.length?<GenerationControl generations={generations} value={selectedGeneration!} onChange={setGeneration}/>:<div className="data-status data-status--missing"><i/>Search generations were not retained for this Sample configuration.</div>}</div>
        <div className="ahd-evidence-column">
          {current?<article className="candidate-card"><div className="pane-head"><div><span className="eyebrow">Search population</span><h3>Generation {selectedGeneration}</h3></div><div className="candidate-best"><small>Best found so far</small><strong>{current.bestSoFar.toFixed(4)}</strong></div></div><div className="candidate-chart-head"><strong>Candidate objective distribution</strong><p>Bar height = number of valid candidate evaluations in that objective range.</p></div><CandidateHistogram values={finite} generation={selectedGeneration!}/><div className="candidate-stats"><span><b>{finite.length}</b>valid candidates</span><span><b>{current.invalidCandidates}</b>invalid / timeout</span><span><b>{current.operator}</b>last operator</span></div></article>:replayCurrent?<article className="candidate-card"><div className="pane-head"><div><span className="eyebrow">Favorable code replay</span><h3>Generation {selectedGeneration}</h3></div><strong>{replayCurrent.objective.toFixed(5)}</strong></div><p className="capability-feedback">{replayCurrent.algorithm}</p><div className="data-status"><i/>Exact heuristic source retained for this generation</div></article>:<article className="candidate-card candidate-card--missing"><span className="eyebrow">Retention status</span><h3>Final artifact only</h3><p>The selected Sample run retained its final best heuristic, but not per-generation candidate objectives or operator history.</p></article>}
          <section className={`ahd-code-panel ahd-code-panel--${displayedHeuristic?"available":"missing"}`}><SectionHeading eyebrow="Heuristic artifact" title={displayedHeuristic?(replayCurrent?`Retained heuristic at generation ${selectedGeneration}`:"The retained final best heuristic"):"Candidate code was not retained for this step"}>{displayedHeuristic?(replayCurrent?"This exact Python function is linked to the selected favorable recheck generation.":"This complete Python function is copied from the selected original result artifact."):`The log retains objectives and operator names at generation ${selectedGeneration}, but not intermediate candidate source.`}</SectionHeading>{displayedHeuristic?<pre className="code-block"><HighlightedCode>{displayedHeuristic}</HighlightedCode></pre>:<div className="missing-trajectory"><span>○</span><strong>Objective retained · code unavailable</strong><p>Move to generation {finalGeneration} to inspect the final best heuristic.</p></div>}</section>
        </div>
      </div>
    </section>
    <section className="section-shell results-section"><SectionHeading eyebrow="Final evaluation" title="Matched EoH comparison on constructive TSP" aside={<BaselineToggle checked={baselines==="on"} onChange={(value)=>setBaselines(value?"on":"off")}/>}/><ResultTable rows={data.finalResults} showBaselines={baselines==="on"} columns={[{key:"budget",label:"Budget"},{key:"method",label:"Method"},{key:"tsp20",label:"TSP · N=20"},{key:"tsp50",label:"TSP · N=50"}]}/><DataProvenance metadata={{...data.metadata,sourceFiles:[...selectedConfig.sourceFiles,"Paper table: AHD constructive results (main.tex)"],note:`Selected ${selectedConfig.problem} ${selectedConfig.mode} · ${selectedConfig.outerMethod} · ${selectedConfig.agenticESOpt?"Agentic ESOpt on":"Agentic ESOpt off"} · budget ${selectedConfig.budget} · repeat ${selectedConfig.repeat}. ${data.metadata.note}`}}/></section>
  </PageFrame>;
}
