import { useEffect, useRef, useState, type CSSProperties } from "react";

import { MethodLoop } from "../components/MethodLoop";
import { CapabilityExplorer } from "../components/CapabilityExplorer";
import { PaperByline } from "../components/PaperByline";
import { HomeScalingSummary } from "../components/ScalingSummary";
import { PageFrame } from "../components/SiteChrome";
import { SectionHeading } from "../components/TaskPrimitives";
import { Tex } from "../components/Tex";
import { paper, paperCitation } from "../paper";
import { siteHref } from "../site";

const tasks = [
  { index: "01", name: "Sudoku", family: "Long-horizon control", comparison: "15-turn success · strongest GRPO → Agentic ESOpt", result: "40.63% → 53.13%", change: "+30.77%", path: "tasks/sudoku/" },
  { index: "02", name: "Math", family: "Reasoning + tools", comparison: "AIME Mean@4 · Agentic GRPO → Agentic ESOpt", result: "58.3% → 70.8%", change: "+21.44%", path: "tasks/math/" },
  { index: "03", name: "DocVQA", family: "Vision + OCR tools", comparison: "Mean@4 accuracy · Agentic GRPO → Agentic ESOpt", result: "48.0% → 52.5%", change: "+9.38%", path: "tasks/docvqa/" },
  { index: "04", name: "WebArena", family: "Web interaction", comparison: "WebArena-Lite · No Skill → Agentic ESOpt", result: "29.47% → 36.16%", change: "+22.70%", path: "tasks/webarena/" },
  { index: "05", name: "AHD", family: "Heuristic design", comparison: "28 of 36 matched Sample / EoH comparisons", result: "77.78%", change: "improved", path: "tasks/ahd/" },
];

type OrbitReward = 1 | 2 | 3 | 4 | 5;

const orbitRewardEncoding: Record<OrbitReward, { color: string; weightedLength: number }> = {
  1: { color: "#aab9a5", weightedLength: 24 },
  2: { color: "#8fb278", weightedLength: 37 },
  3: { color: "#70a864", weightedLength: 50 },
  4: { color: "#489155", weightedLength: 64 },
  5: { color: "#1a6b4a", weightedLength: 77 },
};

const orbitGeometry = {
  currentModel: { x: 315, y: 270, radius: 48 },
  ringRadii: [205, 154, 94],
  glowRadius: 112,
  weightedDirectionStartRadius: 56,
  nextModel: { x: 431, y: 261, radius: 30 },
} as const;

const orbitNodes = ([
  { angle: -144, radius: 205, reward: 2 },
  { angle: -108, radius: 145, reward: 3 },
  { angle: -72, radius: 205, reward: 1 },
  { angle: -36, radius: 145, reward: 5 },
  { angle: 0, radius: 205, reward: 1 },
  { angle: 36, radius: 145, reward: 4 },
  { angle: 72, radius: 205, reward: 2 },
  { angle: 108, radius: 145, reward: 4 },
  { angle: 144, radius: 205, reward: 3 },
  { angle: 180, radius: 145, reward: 2 },
] satisfies ReadonlyArray<{ angle: number; radius: number; reward: OrbitReward }>).map(({ angle, radius, reward }) => {
  const radians = angle * Math.PI / 180;
  return {
    angle,
    reward,
    x: orbitGeometry.currentModel.x + Math.cos(radians) * radius,
    y: orbitGeometry.currentModel.y + Math.sin(radians) * radius,
  };
});

function pointAlongDirection(x: number, y: number, radius: number) {
  const { currentModel } = orbitGeometry;
  const dx = x - currentModel.x;
  const dy = y - currentModel.y;
  const scale = radius / Math.hypot(dx, dy);
  return { x: currentModel.x + dx * scale, y: currentModel.y + dy * scale };
}

function weightedDirection(x: number, y: number, reward: OrbitReward) {
  const { weightedDirectionStartRadius } = orbitGeometry;
  const { weightedLength } = orbitRewardEncoding[reward];
  const guideLength = Math.hypot(x - orbitGeometry.currentModel.x, y - orbitGeometry.currentModel.y) - weightedDirectionStartRadius;
  const start = pointAlongDirection(x, y, weightedDirectionStartRadius);
  const end = pointAlongDirection(x, y, weightedDirectionStartRadius + weightedLength);
  return { start, end, weightedLength, weightFraction: Math.min(weightedLength / guideLength, 1) };
}

const orbitStages = [
  { id: "perturb", index: "01", title: "Perturb", detail: "Sample nearby policies" },
  { id: "evaluate", index: "02", title: "Evaluate", detail: "Run full trajectories" },
  { id: "update", index: "03", title: "Update", detail: "Weight directions · move model" },
] as const;

type OrbitStage = typeof orbitStages[number]["id"];

export function HomePage() {
  const [activeOrbitStage, setActiveOrbitStage] = useState<OrbitStage>("perturb");
  const [orbitInView, setOrbitInView] = useState(true);
  const orbitRef = useRef<HTMLElement>(null);
  const { currentModel, ringRadii, glowRadius, nextModel } = orbitGeometry;
  const resultantArrowStart = pointAlongDirection(nextModel.x, nextModel.y, currentModel.radius + 4);
  const resultantArrowEnd = pointAlongDirection(nextModel.x, nextModel.y, Math.hypot(nextModel.x - currentModel.x, nextModel.y - currentModel.y) - nextModel.radius);

  useEffect(() => {
    const figure = orbitRef.current;
    if (!figure || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver(([entry]) => setOrbitInView(entry.isIntersecting));
    observer.observe(figure);
    return () => observer.disconnect();
  }, []);

  return (
    <PageFrame>
      <section className="home-hero section-shell">
        <div className="hero-copy">
          <div className="paper-tag"><span />Agentic ESOpt</div>
          <h1 className="display-title">
            <span className="hero-title__line">Fine-Tuning </span>
            <span className="hero-title__line hero-title__agents">Long-Horizon LLM Agents <small>with</small>{" "}</span>
            <span className="hero-title__line hero-title__memory"><em>Minimal GPU Memory Requirements.</em></span>
          </h1>
          <p className="hero-deck">
            <span className="hero-deck__lead">Explore <mark>gradient-free</mark> agent fine-tuning: </span>
            <span className="hero-deck__detail">Agentic ESOpt learns from trajectory-level black-box feedback without backpropagation.</span>
          </p>
          <ul className="hero-task-points" aria-label="Agentic task coverage">
            {tasks.map((task) => <li key={task.name}><span><strong>{task.name}</strong><small>{task.family}</small></span></li>)}
          </ul>
          <div className="hero-actions">
            <a className="button button--primary button--featured" href={paper.githubUrl} target="_blank" rel="noreferrer">View code ↗</a>
            <a className="button button--ghost" href={paper.checkpointCollectionUrl} target="_blank" rel="noreferrer">Hugging Face ↗</a>
            <a className="button button--ghost" href={paper.dailyPapersUrl} target="_blank" rel="noreferrer">Browse Daily Papers ↗</a>
          </div>
        </div>
        <figure ref={orbitRef} className={`hero-orbit hero-orbit--${activeOrbitStage}${orbitInView ? "" : " hero-orbit--paused"}`} aria-label="Interactive ES generation stages">
          <div className="hero-orbit__heading"><span>One ES generation</span><strong>Optimize the agent from complete trajectories.</strong><small>Hover, focus, or tap a step to animate it.</small></div>
          <svg viewBox="80 65 520 410" role="img" aria-label="One gradient-free ES generation" aria-describedby="hero-orbit-description">
            <title>One gradient-free ES generation</title>
            <desc id="hero-orbit-description">The current model is perturbed into nearby policies. During evaluation, darker green candidates have higher trajectory rewards. During update, dashed arrows point from the current model toward every sampled direction, with length encoding reward weight; one thick resultant arrow points to the updated model.</desc>
            <defs>
              <radialGradient id="glow"><stop offset="0" stopColor="#d7ff69" stopOpacity=".7"/><stop offset="1" stopColor="#d7ff69" stopOpacity="0"/></radialGradient>
              <linearGradient id="orbit-reward-gradient" x1="0" y1="1" x2="0" y2="0"><stop offset="0" stopColor={orbitRewardEncoding[1].color}/><stop offset=".5" stopColor={orbitRewardEncoding[3].color}/><stop offset="1" stopColor={orbitRewardEncoding[5].color}/></linearGradient>
              {orbitNodes.map(({ x, y, reward }, index) => {
                const direction = weightedDirection(x, y, reward);
                const weightedOffset = `${direction.weightFraction * 100}%`;
                const rewardColor = orbitRewardEncoding[reward].color;
                return <linearGradient id={`orbit-weighted-direction-${index}`} gradientUnits="userSpaceOnUse" x1={direction.start.x} y1={direction.start.y} x2={x} y2={y} key={`${x}-${y}-gradient`}><stop offset="0" stopColor={rewardColor}/><stop offset={weightedOffset} stopColor={rewardColor}/><stop offset={weightedOffset} stopColor="#c8d0c4" stopOpacity=".5"/><stop offset="1" stopColor="#c8d0c4" stopOpacity=".5"/></linearGradient>;
              })}
              <marker id="orbit-update-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path className="orbit-arrow-head" d="M 0 0 L 10 5 L 0 10 z"/></marker>
              <marker id="orbit-contribution-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="4.5" markerHeight="4.5" orient="auto"><path className="orbit-arrow-head" d="M 0 0 L 10 5 L 0 10 z"/></marker>
            </defs>
            {ringRadii.map((radius, index) => <circle cx={currentModel.x} cy={currentModel.y} r={radius} className={`orbit-ring${index === 0 ? " orbit-ring--outer" : ""}`} key={radius}/>)}
            <circle cx={currentModel.x} cy={currentModel.y} r={glowRadius} fill="url(#glow)"/>
            <g className="orbit-reward-scale" aria-hidden="true">
              <rect x="548" y="130" width="50" height="110" rx="8" className="orbit-side-panel"/>
              <text x="573" y="149" textAnchor="middle" className="orbit-reward-scale__title">REWARD</text>
              <text x="573" y="166" textAnchor="middle">HIGH</text>
              <rect x="568" y="174" width="10" height="42" rx="5" fill="url(#orbit-reward-gradient)" className="orbit-reward-bar"/>
              <text x="573" y="231" textAnchor="middle">LOW</text>
            </g>
            <g className="orbit-update-summary" aria-hidden="true">
              <rect x="548" y="130" width="50" height="110" rx="8" className="orbit-side-panel"/>
              <text x="573" y="158" textAnchor="middle" className="orbit-update-summary__title">ALL</text>
              <text x="573" y="175" textAnchor="middle" className="orbit-update-summary__title">SAMPLES</text>
              <line x1="557" y1="188" x2="589" y2="188"/>
              <text x="573" y="207" textAnchor="middle" className="orbit-update-summary__detail">WEIGHTED</text>
              <text x="573" y="221" textAnchor="middle" className="orbit-update-summary__detail">SUM</text>
            </g>
            {orbitNodes.map(({ x, y, angle, reward }, index) => <g className="orbit-node" data-angle={angle} data-reward={reward} style={{ "--orbit-delay": `${index * 70}ms`, "--orbit-reward-color": orbitRewardEncoding[reward].color } as CSSProperties} key={`${x}-${y}`}><line x1={currentModel.x} y1={currentModel.y} x2={x} y2={y}/><circle cx={x} cy={y} r="8"/></g>)}
            <g className="orbit-update-directions" aria-hidden="true">
              {orbitNodes.map(({ x, y, reward }, index) => {
                const direction = weightedDirection(x, y, reward);
                return <polyline points={`${direction.start.x},${direction.start.y} ${direction.end.x},${direction.end.y} ${x},${y}`} data-weight={reward} data-weighted-length={direction.weightedLength} markerMid="url(#orbit-contribution-arrow)" stroke={`url(#orbit-weighted-direction-${index})`} className="orbit-update-weighted-direction" style={{ "--orbit-delay": `${index * 25}ms` } as CSSProperties} key={`${x}-${y}-weighted-direction`}/>;
              })}
            </g>
            <circle cx={currentModel.x} cy={currentModel.y} r={currentModel.radius} className="orbit-core"/>
            <foreignObject x={currentModel.x - 45} y={currentModel.y - 36} width="90" height="36" className="orbit-theta-math" aria-hidden="true"><div><Tex math={"\\theta_t"} label="theta t"/></div></foreignObject>
            <text x={currentModel.x} y={currentModel.y + 12} textAnchor="middle" className="orbit-caption">CURRENT</text><text x={currentModel.x} y={currentModel.y + 28} textAnchor="middle" className="orbit-caption">MODEL</text>
            <g className="orbit-update-flow" aria-hidden="true">
              <path className="orbit-update-resultant" d={`M${resultantArrowStart.x} ${resultantArrowStart.y} L${resultantArrowEnd.x} ${resultantArrowEnd.y}`} markerEnd="url(#orbit-update-arrow)"/>
              <circle cx={nextModel.x} cy={nextModel.y} r={nextModel.radius} className="orbit-next-core"/>
              <foreignObject x={nextModel.x - 38} y={nextModel.y - 15} width="76" height="30" className="orbit-next-theta-math" aria-hidden="true"><div><Tex math={"\\theta_{t+1}"} label="theta t plus 1"/></div></foreignObject>
            </g>
          </svg>
          <figcaption>
            <ol className="orbit-steps" aria-label="One ES generation">
              {orbitStages.map((stage) => <li className={activeOrbitStage === stage.id ? "active" : ""} key={stage.id}><button type="button" aria-pressed={activeOrbitStage === stage.id} onMouseEnter={() => setActiveOrbitStage(stage.id)} onFocus={() => setActiveOrbitStage(stage.id)} onClick={() => setActiveOrbitStage(stage.id)}><b>{stage.index}</b><span><strong>{stage.title}</strong><small>{stage.detail}</small></span></button></li>)}
            </ol>
            <p className="hero-orbit__note"><i />Gradient-free · no backpropagation</p>
          </figcaption>
        </figure>
        <div className="hero-foot"><span>{"Qwen 3.5 · {4B, 9B, 27B}"}</span><span>Five agentic environments</span><span>Inference-level memory</span></div>
      </section>

      <section className="claim-strip"><div><span>01</span><strong>Model Scalability</strong><p>ES enables full-parameter optimization with only minimal, inference-level GPU memory, making it possible to fine-tune large LLMs.</p></div><div><span>02</span><strong>Flexibility</strong><p>Its lightweight, black-box feedback interface makes ES fine-tuning easy to compose with prompt-space evolution (e.g., skill optimization &amp; test-time compute).</p></div><div><span>03</span><strong>Long-Horizon Scalability</strong><p>ES performs trajectory-level parameter attribution without decomposing rewards across horizons, yielding better scalability than Agentic RL as the horizon length grows.</p></div></section>

      <section className="section-shell method-section">
        <SectionHeading eyebrow="The method" title="Full-parameter ES optimization from scalar environment rewards.">Agentic ESOpt samples parameter perturbations around the current LLM, evaluates the perturbed agents with scalar environment rewards, and applies a reward-weighted parameter update.</SectionHeading>
        <MethodLoop />
      </section>

      <section id="explore" className="section-shell environment-results-section">
        <SectionHeading eyebrow="Five environments · result highlights" title="Five environments, each with a matched comparison.">Each row compares Agentic ESOpt with the corresponding base or trained baseline reported in the manuscript; headline performance values use one unit, percent.</SectionHeading>
        <ul className="environment-comparisons" aria-label="Five environment comparisons">
          {tasks.map((task) => <li key={task.name}><a href={siteHref(task.path)}><span className="environment-comparisons__index">{task.index}</span><div className="environment-comparisons__task"><small>{task.family}</small><h3>{task.name}</h3></div><p>{task.comparison}</p><strong>{task.result}</strong><b>{task.change}</b><i aria-hidden="true">↗</i></a></li>)}
        </ul>
      </section>

      <section className="section-shell capability-section capability-section--home">
        <SectionHeading eyebrow="Capability over ES steps" title="Drag the curve. Watch the same case change.">Each tab uses the most ES-favorable eligible case with retained same-case evidence. Switch among Sudoku, WebArena, and automatic heuristic design; every marker is tied to a real result or artifact from the same optimization step.</SectionHeading>
        <CapabilityExplorer />
      </section>

      <section className="section-shell scaling-callout">
        <div><span className="eyebrow">Model-size &amp; ES population scaling</span><h2>How do backbone size and ES population change the result?</h2><p>The experiment provides initial population-sensitivity evidence: doubling G changes final-test success by +677.0% for 4B but 0.0% for 9B. That indicates stronger backbones may need fewer sampled directions.</p><a className="text-link scaling-callout__link" href={siteHref("scaling/")}>Open the scaling matrix →</a></div>
        <div className="scaling-callout__visual">
          <HomeScalingSummary/>
          <p className="scaling-callout__note">Setup: 15-turn Sudoku; Qwen3.5-&#123;4B, 9B&#125;; G ∈ &#123;8, 16&#125;. G is the number of perturbation directions per ES update, not a physical compute-node count.</p>
        </div>
      </section>

      <section className="section-shell home-acknowledgement" aria-labelledby="home-acknowledgement-title">
        <div className="home-acknowledgement__heading"><span className="eyebrow">Research community</span><h2 id="home-acknowledgement-title">Acknowledgement</h2></div>
        <p>{paper.acknowledgement}</p>
      </section>

      <section className="paper-cta"><div className="section-shell"><span className="eyebrow">Read the research</span><h2>{paper.title}</h2><PaperByline/><div><a className="button button--light" href={siteHref("paper/")}>Open paper ↗</a><a className="button button--outline-light" href={paper.githubUrl}>View code ↗</a></div><details className="home-citation"><summary>Citation</summary><pre>{paperCitation}</pre></details></div></section>
    </PageFrame>
  );
}
