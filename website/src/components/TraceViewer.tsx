import { useEffect, useState } from "react";
import type { CaseCheckpoint, ReactStep } from "../types";
import { GenerationControl } from "./GenerationControl";
import { HighlightedCode } from "./HighlightedCode";

function classifyStep(step: ReactStep) {
  if (step.action) return "Tool action";
  if (step.assistant.toLowerCase().includes("final answer")) return "Final answer";
  return "Model response";
}

export function TraceViewer({ checkpoint, turn, onTurnChange }: { checkpoint: CaseCheckpoint; turn: number; onTurnChange: (turn: number) => void }) {
  const [showReasoning, setShowReasoning] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [expandObservation, setExpandObservation] = useState(false);
  const turns = checkpoint.steps.map((step) => step.turn);
  const safeTurn = turns.includes(turn) ? turn : turns[0] ?? 1;
  const step = checkpoint.steps.find((item) => item.turn === safeTurn);
  useEffect(() => {
    setShowReasoning(false);
    setShowRaw(false);
    setExpandObservation(false);
  }, [checkpoint.generation, safeTurn]);
  if (!step) return <div className="empty-state">Trajectory not recorded for this checkpoint.</div>;
  return (
    <div className="trace-viewer">
      {turns.length > 1
        ? <GenerationControl generations={turns} value={safeTurn} onChange={onTurnChange} label="ReAct turn" />
        : <div className="trace-single-turn" role="status"><span>ReAct turn</span><strong>Single retained ReAct turn</strong><small>Turn 1 of 1</small></div>}
      <div className="trace-step">
        <div className="trace-step__head"><span className="trace-kind">{classifyStep(step)}</span><span>Turn {step.turn} of {turns.length}</span></div>
        <div className="trace-disclosure">
          <button type="button" aria-pressed={showReasoning} onClick={() => setShowReasoning((value) => !value)}>{showReasoning ? "Hide reasoning" : "Show reasoning"}</button>
          <button type="button" aria-pressed={showRaw} onClick={() => setShowRaw((value) => !value)}>{showRaw ? "Hide raw trace" : "Show raw trace"}</button>
          {step.observation && <button type="button" aria-pressed={expandObservation} onClick={() => setExpandObservation((value) => !value)}>{expandObservation ? "Collapse observation" : "Expand observation"}</button>}
        </div>
        {step.action && <section><span className="trace-label">Action · {step.action.name || "tool"}</span><pre><HighlightedCode>{step.action.arguments?.command || step.assistant}</HighlightedCode></pre></section>}
        {showReasoning && <section className="reasoning"><span className="trace-label">Model response</span><p>{step.assistant}</p></section>}
        {step.observation && <section className={`observation ${expandObservation ? "expanded" : ""}`}><span className="trace-label">Observation</span><pre>{step.observation}</pre></section>}
        {showRaw && <section className="raw-trace"><span className="trace-label">Raw retained step</span><pre><HighlightedCode>{JSON.stringify(step, null, 2)}</HighlightedCode></pre></section>}
      </div>
      <div className="trace-outcome"><div><span>Prediction</span><strong>{checkpoint.prediction || "No parsed answer"}</strong></div><div><span>Trajectory score</span><strong className={checkpoint.score > 0.5 ? "success-text" : "failure-text"}>{checkpoint.score.toFixed(2)}</strong></div><div><span>Termination</span><strong>{checkpoint.terminationReason.replaceAll("_", " ") || "unknown"}</strong></div></div>
    </div>
  );
}
