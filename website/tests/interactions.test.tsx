import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { GenerationControl } from "../src/components/GenerationControl";
import { CapabilityExplorerView, CapabilityPanel, diffLines } from "../src/components/CapabilityExplorer";
import { MethodLoop } from "../src/components/MethodLoop";
import { MetricChart } from "../src/components/MetricChart";
import { ScalingSummary } from "../src/components/ScalingSummary";
import { SiteHeader } from "../src/components/SiteChrome";
import { TaskHero } from "../src/components/TaskPrimitives";
import { SudokuBoard } from "../src/components/SudokuBoard";
import { TraceViewer } from "../src/components/TraceViewer";
import { HomePage } from "../src/pages/HomePage";
import { AhdPage } from "../src/pages/AhdPage";
import { DocVqaPage } from "../src/pages/DocVqaPage";
import { mathGenerationOptions } from "../src/pages/MathPage";
import { PaperPage } from "../src/pages/PaperPage";
import { ScalingExplorer, type ScaleResult } from "../src/pages/ScalingPage";
import { SudokuPage } from "../src/pages/SudokuPage";
import type { TaskPayload } from "../src/types";
import sudokuData from "../public/data/sudoku.json";
import docvqaData from "../public/data/docvqa.json";
import ahdData from "../public/data/ahd.json";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState(null, "", "/");
});

describe("research interactions", () => {
  it("explains the AHD candidate histogram and omits an instance without a retained route", async () => {
    window.history.replaceState(null, "", "/?problem=TSP&mode=Constructive&outer=EoH&agentic=off&budget=1000&repeat=1&generation=1");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ahdData }));
    const { container } = render(<AhdPage />);

    await screen.findByRole("heading", { name: "Generation 1" });
    expect(screen.getByText("Bar height = number of valid candidate evaluations in that objective range.")).toBeInTheDocument();
    expect(screen.getByText("Lower internal objective · better")).toBeInTheDocument();
    expect(screen.getByText("Higher internal objective · worse")).toBeInTheDocument();
    const histogram = screen.getByRole("img", { name: /Candidate objective histogram for generation 1/ });
    expect(histogram).toHaveAccessibleName(/Bin counts from lower to higher objective: 7\.00 to 9\.97: 9;.*33\.68 to 36\.65: 2/);
    const bins = Array.from(container.querySelectorAll<HTMLElement>(".candidate-histogram__bin"));
    expect(bins).toHaveLength(10);
    expect(bins.reduce((sum, bin) => sum + Number(bin.dataset.count), 0)).toBe(25);
    expect(bins.at(-1)).toHaveAttribute("data-count", "2");

    expect(screen.queryByText("50-city TSP input instance")).not.toBeInTheDocument();
    expect(screen.queryByText("Recorded route unavailable")).not.toBeInTheDocument();
    expect(screen.queryByRole("img", { name: /Fifty-city TSP/ })).not.toBeInTheDocument();
  });

  it("stacks the retained heuristic directly beneath Search population", async () => {
    window.history.replaceState(null, "", "/?problem=TSP&mode=Constructive&outer=EoH&agentic=off&budget=1000&repeat=1&generation=25");
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => ahdData }));
    render(<AhdPage />);

    const populationHeading = await screen.findByRole("heading", { name: "Generation 25" });
    const codeHeading = screen.getByRole("heading", { name: "The retained final best heuristic" });
    const populationCard = populationHeading.closest("article");
    const codePanel = codeHeading.closest("section");
    expect(populationCard?.parentElement).toBe(codePanel?.parentElement);
    expect(populationCard?.compareDocumentPosition(codePanel!)).toBe(Node.DOCUMENT_POSITION_FOLLOWING);
    expect(codePanel).toHaveClass("ahd-code-panel--available");
  });

  it("places the DocVQA curve split control beside the chart it changes", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => docvqaData }));
    const { container } = render(<DocVqaPage />);
    await screen.findByRole("heading", { name: "Held-out ANLS" });
    const curveSection = container.querySelector<HTMLElement>(".curve-section")!;
    const caseControl = container.querySelector<HTMLElement>(".doc-case-control")!;
    const splitControl = within(curveSection).getByRole("group", { name: "Curve split" });
    expect(within(caseControl).queryByRole("group", { name: "Curve split" })).not.toBeInTheDocument();
    expect(curveSection.querySelectorAll(".series-dot")).toHaveLength(5);
    fireEvent.click(within(splitControl).getByRole("button", { name: "Train" }));
    expect(within(curveSection).getByRole("heading", { name: "Training reward" })).toBeInTheDocument();
    expect(curveSection.querySelectorAll(".series-dot")).toHaveLength(40);
  });

  it("uses the task-level average success rate as the Sudoku hero metric", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true, json: async () => sudokuData }));
    const { container } = render(<SudokuPage />);
    const heroMetric = await screen.findByText("Average final-test success rate");
    const metricCard = container.querySelector(".hero-metric");
    expect(metricCard).toContainElement(heroMetric);
    expect(metricCard).toHaveTextContent("89.58%");
    expect(metricCard).toHaveTextContent("Agentic ESOpt · mask 5 · task-level average");
    expect(metricCard).not.toHaveTextContent("0% → 100%");
  });

  it("moves a fixed Sudoku case and its curve through capability checkpoints", () => {
    const data = {
      metadata: { task: "sudoku", method: "Agentic ESOpt", sourceFiles: ["history"] },
      configurations: [],
      curves: [{
        id: "mask5-stage3-recheck-eval",
        configId: "sudoku-mask5-stage3-recheck",
        label: "Stage 3 recheck · periodic evaluation",
        kind: "eval",
        points: [{ generation: -1, value: 0.25 }, { generation: 39, value: 0.9 }],
      }],
      checkpoints: [],
      cases: [{
        id: "eval-1",
        maskCount: 5,
        puzzle: [[1, 0], [0, 1]],
        solution: [[1, 2], [2, 1]],
        evidenceScope: "Accepted Stage 3 recheck",
        capabilityCheckpoints: [
          { optimizationStep: -1, aggregateMetric: 0.25, score: 0, prediction: [[1, 1], [2, 1]], feedback: "invalid row", turns: [] },
          { optimizationStep: 39, aggregateMetric: 0.9, score: 1, prediction: [[1, 2], [2, 1]], feedback: "valid", turns: [] },
        ],
      }],
      finalResults: [],
    } as unknown as TaskPayload;

    render(<CapabilityPanel task="sudoku" data={data} />);
    expect(screen.getByText("Failure")).toBeInTheDocument();
    expect(screen.getByText("25.0%")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Next generation" }));
    expect(screen.getByText("Success")).toBeInTheDocument();
    expect(screen.getByText("90.0%")).toBeInTheDocument();
    expect(screen.getByLabelText("Selected generation 39")).toBeInTheDocument();
    expect(document.querySelector(".capability-artifact__step")).toHaveTextContent("ES generation39");
    expect(screen.getByText(/Solved: 2\/2 masked cells/)).toBeInTheDocument();
    expect(document.querySelectorAll('[data-changed="true"]')).toHaveLength(1);
  });

  it("opens WebArena on its retained output and uses compact score evidence for earlier epochs", () => {
    const data = {
      metadata: { task: "webarena", method: "Agentic ESOpt", sourceFiles: [] }, configurations: [], checkpoints: [], finalResults: [],
      curves: [{ id: "esopt-eval", label: "Periodic evaluation", kind: "eval", points: [{ generation: 10, value: 0.2 }, { generation: 50, value: 0.35 }, { generation: 70, value: 0.4 }] }],
      cases: [{ id: "task-4", goal: "List fingerprint-resistant reviewers", evidenceScope: "Favorable eligible case", capabilityCheckpoints: [
        { optimizationStep: 10, aggregateMetric: 0.2, score: 0, outputUnavailable: true },
        { optimizationStep: 50, aggregateMetric: 0.35, score: 1, outputUnavailable: true },
        { optimizationStep: 70, aggregateMetric: 0.4, score: 1, output: "Reviewer A, Reviewer B", outputUnavailable: false },
      ] }],
    } as unknown as TaskPayload;
    window.history.replaceState({}, "", "/?cap_webarena_step=999");
    render(<CapabilityPanel task="webarena" data={data}/>);
    expect(screen.getByText("Reviewer A, Reviewer B")).toBeInTheDocument();
    expect(screen.getByLabelText("Selected evaluation epoch 70")).toBeInTheDocument();
    expect(screen.queryByText("Readable output was not retained at this epoch")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Early 10/ }));
    expect(screen.getByText("Score-only checkpoint")).toBeInTheDocument();
    expect(screen.getByText("0 / 1")).toBeInTheDocument();
    expect(screen.getByText(/Still failing/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Middle 50/ }));
    expect(screen.getByText(/case score changed 0 → 1/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /Late 70/ }));
    expect(screen.getByText(/case score remains 1 → 1/)).toBeInTheDocument();
    expect(window.location.search).toContain("cap_webarena_step=70");
  });

  it("highlights exact AHD heuristic changes between retained versions", () => {
    const data = {
      metadata: { task: "ahd", method: "Agentic ESOpt", sourceFiles: [] }, configurations: [], checkpoints: [], finalResults: [],
      curves: [{ id: "ahd-cap", label: "Best objective", kind: "train", capabilityCurve: true, points: [{ generation: 1, value: 8 }, { generation: 4, value: 7 }, { generation: 25, value: 6 }] }],
      cases: [{ id: "tsp", evidenceScope: "Accepted recheck", capabilityCheckpoints: [
        { optimizationStep: 1, objective: 8, heuristic: "def h():\n    return 1", algorithm: "v1" },
        { optimizationStep: 4, objective: 7, heuristic: "def h():\n    return 2", algorithm: "v2" },
        { optimizationStep: 25, objective: 6, heuristic: "def h():\n    return 3", algorithm: "v3", testInstanceMinimum: { value: 5.3, scope: "TSP-50 · minimum across 64 frozen-test instances" } },
      ] }],
    } as unknown as TaskPayload;
    const { container } = render(<CapabilityPanel task="ahd" data={data}/>);
    expect(screen.getAllByText("6.00000")).toHaveLength(2);
    expect(screen.queryByText("5.30000")).not.toBeInTheDocument();
    expect(screen.queryByText("Frozen-test single-instance minimum")).not.toBeInTheDocument();
    expect(screen.queryByText("TSP-50 · minimum across 64 frozen-test instances")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Previous generation" }));
    expect(screen.queryByText("5.30000")).not.toBeInTheDocument();
    expect(screen.getByText(/Improved the objective by 1.00000/)).toBeInTheDocument();
    expect(screen.getByText("Code changes from previous checkpoint")).toBeInTheDocument();
    expect(container.querySelector(".diff-line--removed")).toHaveTextContent("return 1");
    expect(container.querySelector(".diff-line--added")).toHaveTextContent("return 2");
    expect(screen.getByLabelText("Code changes from previous checkpoint")).toHaveAttribute("tabindex", "0");
    expect(screen.getByLabelText("Retained heuristic source code")).toHaveAttribute("tabindex", "0");
  });

  it("persists capability task selection and isolates a failed background task", () => {
    const sudoku = {
      metadata: { task: "sudoku", method: "Agentic ESOpt", sourceFiles: [] }, configurations: [], checkpoints: [], finalResults: [],
      curves: [{ id: "mask5-stage3-recheck-eval", label: "Eval", kind: "eval", points: [{ generation: -1, value: 0.1 }] }],
      cases: [{ id: "s", puzzle: [[0]], solution: [[1]], evidenceScope: "recheck", capabilityCheckpoints: [{ optimizationStep: -1, aggregateMetric: 0.1, score: 0, prediction: [[0]] }] }],
    } as unknown as TaskPayload;
    render(<CapabilityExplorerView resources={{ sudoku: { data: sudoku, error: "" }, webarena: { data: null, error: "failed" }, ahd: { data: null, error: "" } }}/>);
    expect(screen.getByText("s")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open full Sudoku explorer →" })).toHaveAttribute("href", "/tasks/sudoku/");
    expect(screen.queryByText(/could not be loaded/)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: /WebArena web outcome/ }));
    expect(screen.getByText("WebArena capability data could not be loaded.")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Open full WebArena explorer →" })).toHaveAttribute("href", "/tasks/webarena/");
    expect(window.location.search).toContain("cap_task=webarena");
  });

  it("computes a stable line diff for AHD code", () => {
    expect(diffLines("a\nb", "a\nc")).toEqual([
      { kind: "same", text: "a" },
      { kind: "removed", text: "b" },
      { kind: "added", text: "c" },
    ]);
  });

  it("compares model size and ES population as two explicit scaling axes", () => {
    const rows: ScaleResult[] = [
      { model: "4B", population: 8, best: 5.10, final: 2.95 },
      { model: "4B", population: 16, best: 35.42, final: 22.92, finalRelative: 677.0 },
      { model: "9B", population: 8, best: 30.21, final: 30.21 },
      { model: "9B", population: 16, best: 37.50, final: 30.21, finalRelative: 0.0 },
    ];

    render(<ScalingExplorer rows={rows}/>);

    const scalingTitle = screen.getByRole("heading", { level: 1, name: "Scaling across model size and ES population" });
    expect(scalingTitle).toHaveClass("display-title");
    expect(scalingTitle.querySelector(".scaling-hero__axes")).toHaveTextContent("model size and ES population");
    expect(screen.getByText(/Compare Qwen3\.5-4B and 9B at G=8 and G=16/)).toHaveClass("scaling-hero__description");
    expect(screen.getByText("Model size", { selector: ".matrix-axis--side" })).toBeInTheDocument();
    expect(screen.getByText("ES population size", { selector: ".matrix-axis--top" })).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /model, ES population/ })).toHaveLength(4);
    expect(screen.getByText("+2.08 points vs 4B at fixed G=16")).toBeInTheDocument();
    expect(screen.getByText("+7.29 points vs G=8 at fixed 9B")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Final checkpoint" }));
    expect(screen.getByText("+7.29 points vs 4B at fixed G=16")).toBeInTheDocument();
    expect(screen.getByText("+0.00 points vs G=8 at fixed 9B")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /^4B model, ES population 8\./ }));
    expect(screen.getByText("4B · G=8")).toBeInTheDocument();
    expect(window.location.search).toContain("model=4B");
    expect(window.location.search).toContain("population=8");
  });

  it("uses the homepage display-title language for task-page headings", () => {
    render(<TaskHero kicker="Task" title="Tool-Using" accent="Math Reasoning" summary="Summary" metric="Metric" value="1" detail="Detail" />);
    const heading = screen.getByRole("heading", { level: 1, name: "Tool-Using Math Reasoning" });
    expect(heading).toHaveClass("display-title");
    expect(heading.querySelector("em")).toHaveTextContent("Math Reasoning");
  });

  it("keeps scaling out of primary navigation", () => {
    render(<SiteHeader />);
    expect(screen.queryByRole("link", { name: "Scaling" })).not.toBeInTheDocument();
    const mobileMenu = document.querySelector(".mobile-task-menu");
    expect(mobileMenu).not.toBeNull();
    expect(within(mobileMenu as HTMLElement).getByText("More")).toBeInTheDocument();
    expect(within(mobileMenu as HTMLElement).getByRole("link", { name: "Paper" })).toHaveAttribute("href", expect.stringMatching(/paper\/$/));
    const resources = screen.getByRole("group", { name: "Research links" });
    const huggingFaceLink = within(resources).getByRole("link", { name: "Hugging Face ↗" });
    expect(huggingFaceLink).toHaveTextContent(/^Hugging Face ↗$/);
    expect(huggingFaceLink).toHaveAttribute("href", "https://huggingface.co/collections/zz1358m/agentic-esopt-checkpoints-collection-6a781bb727d86d7742e61df6");
    expect(within(resources).getByRole("link", { name: "Code ↗" })).toHaveAttribute("href", "https://github.com/zz1358m/Agentic-ESOpt");
  });

  it("keeps scaling values accessible and rejects incomplete matrices", () => {
    const rows = [
      { model: "4B" as const, population: 8 as const, best: 5.10, final: 2.95 },
      { model: "4B" as const, population: 16 as const, best: 35.42, final: 22.92 },
      { model: "9B" as const, population: 8 as const, best: 30.21, final: 30.21 },
      { model: "9B" as const, population: 16 as const, best: 37.50, final: 30.21 },
    ];
    const { rerender } = render(<ScalingExplorer rows={rows}/>);
    expect(screen.getByRole("button", { name: "9B model, ES population 16. Best checkpoint success 37.50%. Final checkpoint success 30.21%." })).toBeInTheDocument();

    rerender(<ScalingExplorer rows={rows.slice(0, 3)}/>);
    expect(screen.getByText("The complete 4B/9B × G=8/16 scaling matrix is unavailable.")).toBeInTheDocument();

    rerender(<ScalingExplorer rows={[rows[0], rows[0], rows[2], rows[3]]}/>);
    expect(screen.getByText("The complete 4B/9B × G=8/16 scaling matrix is unavailable.")).toBeInTheDocument();
  });

  it("renders the home scaling summary from the shared result rows", () => {
    const { container } = render(<ScalingSummary rows={[
      { model: "4B", population: 8, best: 5.10, final: 2.95 },
      { model: "4B", population: 16, best: 35.42, final: 22.92, finalRelative: 677.0 },
      { model: "9B", population: 8, best: 30.21, final: 30.21 },
      { model: "9B", population: 16, best: 37.50, final: 30.21, finalRelative: 0.0 },
    ]}/>);
    expect(screen.getByLabelText("Model-size and ES population scaling matrix")).toHaveTextContent("4B · G=8");
    expect(screen.getByLabelText("Model-size and ES population scaling matrix")).toHaveTextContent("best 37.50%");
    expect(screen.getByLabelText("Population sensitivity conclusion")).toHaveTextContent("4B+677.0%final success when G doubles9B0.0%");
    expect(container.querySelector(".scale-sensitivity__emphasis")).toHaveTextContent("4B+677.0%");
    expect(container.querySelector(".scale-matrix .active")).not.toBeInTheDocument();
  });

  it("moves the curve cursor with the selected ES generation", () => {
    const { rerender } = render(
      <MetricChart
        series={[{ id: "train", label: "Train reward", kind: "train", points: [{ generation: 0, value: 0.2 }, { generation: 10, value: 0.8 }] }]}
        generation={0}
        title="Training signal"
      />,
    );
    expect(screen.getByLabelText("Selected generation 0")).toBeInTheDocument();
    expect(screen.getByLabelText("Current metrics")).toHaveTextContent("0.20");
    expect(screen.getByLabelText("Current metrics")).toHaveTextContent("+0.00 from initial");

    rerender(
      <MetricChart
        series={[{ id: "train", label: "Train reward", kind: "train", points: [{ generation: 0, value: 0.2 }, { generation: 10, value: 0.8 }] }]}
        generation={10}
        title="Training signal"
      />,
    );
    expect(screen.getByLabelText("Selected generation 10")).toBeInTheDocument();
    expect(screen.getByLabelText("Current metrics")).toHaveTextContent("+0.60 from initial");
  });

  it("selects only generations actually represented by the control", () => {
    const onChange = vi.fn();
    render(<GenerationControl generations={[-1, 9, 19]} value={9} onChange={onChange} />);
    fireEvent.click(screen.getByRole("button", { name: "Next generation" }));
    expect(onChange).toHaveBeenCalledWith(19);
  });

  it("limits visible generation labels so dense runs do not overlap", () => {
    const { container } = render(
      <GenerationControl generations={Array.from({ length: 100 }, (_, generation) => generation)} value={99} onChange={vi.fn()} />,
    );
    const labels = [...container.querySelectorAll(".generation-control__ticks span")].map((item) => item.textContent);
    expect(labels).toEqual(["0", "20", "40", "59", "79", "99"]);
  });

  it("keeps Sudoku givens fixed and marks a conflicting user entry", () => {
    const puzzle = Array.from({ length: 9 }, () => Array(9).fill(0));
    const solution = Array.from({ length: 9 }, () => Array(9).fill(1));
    puzzle[0][0] = 5;
    solution[0][0] = 5;
    solution[0][1] = 3;
    render(<SudokuBoard puzzle={puzzle} solution={solution} resetKey="mask-5" />);

    expect(screen.getByLabelText("Row 1 column 1, given 5")).toBeDisabled();
    const editable = screen.getByLabelText("Row 1 column 2, empty");
    fireEvent.change(editable, { target: { value: "4" } });
    expect(editable).toHaveAttribute("data-state", "conflict");
  });

  it("keeps a replay-only final Math generation reachable", () => {
    const payload = {
      curves: [{ id: "train", label: "Train", kind: "train", points: [{ generation: 0, value: 0 }, { generation: 24, value: 1 }] }],
      cases: [{ checkpoints: [{ generation: 25 }] }],
    } as unknown as TaskPayload;
    expect(mathGenerationOptions(payload, "Train")).toEqual([0, 24]);
    payload.curves.push({ id: "dapo_eval", label: "Eval", kind: "eval", points: [{ generation: 25, value: 1 }] });
    expect(mathGenerationOptions(payload, "Periodic Evaluation", "DAPO")).toEqual([25]);
  });

  it("presents the complete seven-step ES optimization loop", () => {
    const { container } = render(<MethodLoop />);
    expect(screen.getByRole("list", { name: "Agentic ESOpt optimization steps" })).toBeInTheDocument();
    for (const step of ["Current model", "Sample perturbations", "Evaluate agents", "Collect rewards", "Normalize rewards", "Reward-weighted update", "Next generation"]) expect(screen.getByText(step)).toBeInTheDocument();
    expect(screen.getByText("Sample G full-parameter perturbations around the current LLM.")).toBeInTheDocument();
    expect(screen.getByText("Obtain one scalar trajectory reward Rᵢ for each perturbed agent.")).toBeInTheDocument();
    expect(screen.getByText("Normalize rewards within the population using a z-score.")).toBeInTheDocument();
    expect(screen.getByLabelText("theta t plus 1")).toBeInTheDocument();
    expect(screen.getByLabelText("theta t")).toHaveTextContent("θt");
    expect(screen.getByLabelText("theta t plus 1")).toHaveTextContent("θt+1");
    expect(container.querySelectorAll(".method-equation .katex")).toHaveLength(3);
    expect(container.querySelector(".katex-mathml")).not.toBeInTheDocument();
    expect(screen.getByLabelText("theta i equals theta t plus sigma epsilon i")).toBeInTheDocument();
    expect(screen.getByLabelText("reward i equals reward of trajectory i")).toBeInTheDocument();
    expect(screen.getByLabelText("theta t plus 1 equals theta t plus alpha over G times the sum of normalized reward i times epsilon i")).toBeInTheDocument();
    expect(screen.queryByRole("combobox", { name: "Environment" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Play method animation" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Next method step" })).not.toBeInTheDocument();
  });

  it("keeps reasoning and raw retained traces behind explicit controls", () => {
    const checkpoint = {
      generation: 9,
      score: 1,
      prediction: "42",
      terminationReason: "final_answer",
      steps: [{ turn: 1, assistant: "private reasoning text", observation: "long tool output", action: { name: "bash", arguments: { command: "python solve.py" } } }],
    };
    const { container } = render(<TraceViewer checkpoint={checkpoint} turn={1} onTurnChange={vi.fn()} />);
    expect(container.querySelector("pre code")).toBeInTheDocument();
    expect(screen.queryByText("private reasoning text")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show reasoning" }));
    expect(screen.getByText("private reasoning text")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show raw trace" }));
    expect(screen.getByText("Raw retained step")).toBeInTheDocument();
  });

  it("shows a turn slider only when a retained trace has multiple turns", () => {
    const singleTurn = {
      generation: 39,
      score: 1,
      prediction: "Round-Robin Tennis Match",
      terminationReason: "final_answer",
      steps: [{ turn: 1, assistant: "Final answer", observation: "" }],
    };
    const { rerender } = render(<TraceViewer checkpoint={singleTurn} turn={1} onTurnChange={vi.fn()} />);
    expect(screen.queryByRole("slider", { name: "ReAct turn" })).not.toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Single retained ReAct turn");

    const multipleTurns = {
      ...singleTurn,
      generation: 19,
      steps: [
        { turn: 1, assistant: "Inspect document", observation: "OCR text" },
        { turn: 2, assistant: "Final answer", observation: "" },
      ],
    };
    rerender(<TraceViewer checkpoint={multipleTurns} turn={1} onTurnChange={vi.fn()} />);
    expect(screen.getByRole("slider", { name: "ReAct turn" })).toHaveAttribute("max", "1");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("shows the current manuscript metadata on the paper page", () => {
    render(<PaperPage />);

    const paperTitle = screen.getByRole("heading", {
      level: 1,
      name: "Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements",
    });
    expect(paperTitle).toBeInTheDocument();
    expect(paperTitle).toHaveClass("display-title");
    expect(paperTitle.querySelectorAll(".hero-title__line")).toHaveLength(3);
    expect(paperTitle.querySelector(".hero-title__agents small")).toHaveTextContent("with");
    expect(paperTitle.querySelector(".hero-title__memory em")).toHaveTextContent("Minimal GPU Memory Requirements.");
    expect(document.querySelectorAll(".paper-point")).toHaveLength(3);
    expect(screen.getByRole("group", { name: "Paper authors and affiliations" })).toBeInTheDocument();
    expect(document.querySelector(".paper-byline__authors")).toHaveTextContent("Zhi Zheng1 · Rongsheng Chen2 · Yunpeng Ba2 · Zhenkun Wang2 · Yee Whye Teh3 · Wee Sun Lee1");
    expect(screen.getByRole("listitem", { name: "1 National University of Singapore" })).toBeInTheDocument();
    expect(screen.getByRole("listitem", { name: "2 Southern University of Science and Technology" })).toBeInTheDocument();
    expect(screen.getByRole("listitem", { name: "3 Oxford" })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "zhi.zheng@u.nus.edu" })).toHaveAttribute("href", "mailto:zhi.zheng@u.nus.edu");
    expect(screen.getByText(/title = \{Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements\}/)).toBeInTheDocument();
  });

  it("uses the current paper title in the home-page research callout", () => {
    render(<HomePage />);
    const wordmark = screen.getByRole("link", { name: "Agentic ESOpt home" });
    expect(wordmark).toHaveTextContent(/^Agentic ESOpt$/);
    expect(wordmark.querySelector(".wordmark-mark")).not.toBeInTheDocument();
    const heroTitle = screen.getByRole("heading", { level: 1 });
    expect(heroTitle).toHaveClass("display-title");
    expect(heroTitle).toHaveTextContent("Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements");
    expect(heroTitle.querySelectorAll(".hero-title__line")).toHaveLength(3);
    expect(heroTitle.querySelector(".hero-title__agents")).toHaveTextContent("Long-Horizon LLM Agents with");
    expect(heroTitle.querySelector(".hero-title__memory")).toHaveTextContent(/^Minimal GPU Memory Requirements\.$/);
    expect(screen.getByText("gradient-free").tagName).toBe("MARK");
    expect(document.querySelector(".hero-deck__lead")).toHaveTextContent("Explore gradient-free agent fine-tuning:");
    expect(document.querySelector(".hero-deck__detail")).toHaveTextContent("Agentic ESOpt learns from trajectory-level black-box feedback without backpropagation");
    expect(screen.getByText(/without backpropagation/i)).toBeInTheDocument();
    expect(screen.getByText(/most ES-favorable eligible case/i)).toBeInTheDocument();
    const heroActions = document.querySelector(".hero-actions");
    expect(heroActions).not.toBeNull();
    expect(within(heroActions as HTMLElement).queryByRole("link", { name: /Explore tasks/i })).not.toBeInTheDocument();
    expect(within(heroActions as HTMLElement).queryByRole("link", { name: /Read the paper/i })).not.toBeInTheDocument();
    expect(within(heroActions as HTMLElement).getByRole("link", { name: "View code ↗" })).toHaveClass("button--featured");
    expect(within(heroActions as HTMLElement).getByRole("link", { name: "Hugging Face ↗" })).toHaveAttribute("href", "https://huggingface.co/collections/zz1358m/agentic-esopt-checkpoints-collection-6a781bb727d86d7742e61df6");
    expect(within(heroActions as HTMLElement).queryByText(/checkpoints/i)).not.toBeInTheDocument();
    expect(within(heroActions as HTMLElement).getByRole("link", { name: "Browse Daily Papers ↗" })).toHaveAttribute("href", "https://huggingface.co/papers");
    expect(screen.queryByText("Start exploring →")).not.toBeInTheDocument();
    const acknowledgement = screen.getByRole("heading", { level: 2, name: "Acknowledgement" });
    const acknowledgementHeading = acknowledgement.closest(".home-acknowledgement__heading") as HTMLElement;
    expect(within(acknowledgementHeading).getByText("Research community")).toBeInTheDocument();
    expect(acknowledgementHeading).toContainElement(acknowledgement);
    expect(screen.getByText("We would like to sincerely thank Jiaying Wu, Penghui Qi, Zichen Liu, and Ziqiao Meng from the National University of Singapore, as well as Zi'ang Li from Human& for their important comments on the methodology and paper-writing.")).toBeInTheDocument();
    expect(screen.getByText("Qwen 3.5 · {4B, 9B, 27B}")).toBeInTheDocument();
    const taskCoverage = screen.getByRole("list", { name: "Agentic task coverage" });
    expect(within(taskCoverage).getAllByRole("listitem")).toHaveLength(5);
    for (const task of ["Sudoku", "Math", "DocVQA", "WebArena", "AHD"]) {
      expect(within(taskCoverage).getByText(task)).toBeInTheDocument();
    }
    expect(screen.getByRole("img", { name: "One gradient-free ES generation" })).toBeInTheDocument();
    const orbitSteps = screen.getByRole("list", { name: "One ES generation" });
    expect(within(orbitSteps).getByText("Perturb")).toBeInTheDocument();
    expect(within(orbitSteps).getByText("Evaluate")).toBeInTheDocument();
    expect(within(orbitSteps).getByText("Update")).toBeInTheDocument();
    const orbitFigure = screen.getByRole("figure", { name: "Interactive ES generation stages" });
    expect(orbitFigure.querySelector("svg")).toHaveAttribute("viewBox", "80 65 520 410");
    expect(orbitFigure.querySelector(".orbit-ring--outer")).toHaveAttribute("r", "205");
    expect(orbitFigure.querySelector(".orbit-core")).toHaveAttribute("r", "48");
    expect(Array.from(orbitFigure.querySelectorAll(".orbit-caption"), (label) => label.textContent)).toEqual(["CURRENT", "MODEL"]);
    expect(orbitFigure.querySelector(".orbit-theta-math .katex")).toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-next-theta-math .katex")).toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-theta-math .tex")).toHaveTextContent("θt");
    expect(orbitFigure.querySelector(".orbit-next-theta-math .tex")).toHaveTextContent("θt+1");
    expect(orbitFigure.querySelectorAll(".orbit-node[data-reward]")).toHaveLength(10);
    const sampledAngles = Array.from(orbitFigure.querySelectorAll<SVGGElement>(".orbit-node[data-angle]"), (node) => Number(node.dataset.angle)).sort((a, b) => a - b);
    const angularGaps = sampledAngles.map((angle, index) => ((sampledAngles[(index + 1) % sampledAngles.length] ?? angle) - angle + 360) % 360);
    expect(Math.min(...angularGaps)).toBeGreaterThanOrEqual(35);
    expect(orbitFigure.querySelector(".orbit-reward-scale")).toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-reward-scale .orbit-side-panel")).toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-reward-scale .orbit-side-panel")).toHaveAttribute("x", "548");
    expect(orbitFigure.querySelector(".orbit-reward-scale .orbit-side-panel")).toHaveAttribute("width", "50");
    expect(orbitFigure.querySelector(".orbit-reward-bar")).toHaveAttribute("width", "10");
    expect(orbitFigure.querySelector(".orbit-reward-bar")).toHaveAttribute("height", "42");
    expect(orbitFigure.querySelector("#orbit-update-arrow")).toHaveAttribute("markerWidth", "5");
    expect(orbitFigure.querySelector("#orbit-contribution-arrow")).toHaveAttribute("markerWidth", "4.5");
    expect(orbitFigure.querySelector(".orbit-evaluation-flow")).not.toBeInTheDocument();
    const evaluateStep = within(orbitSteps).getByRole("button", { name: /Evaluate/ });
    const updateStep = within(orbitSteps).getByRole("button", { name: /Update/ });
    expect(orbitFigure).toHaveClass("hero-orbit--perturb");
    fireEvent.mouseEnter(evaluateStep);
    expect(orbitFigure).toHaveClass("hero-orbit--evaluate");
    expect(evaluateStep).toHaveAttribute("aria-pressed", "true");
    fireEvent.focus(updateStep);
    expect(orbitFigure).toHaveClass("hero-orbit--update");
    expect(updateStep).toHaveAttribute("aria-pressed", "true");
    expect(within(orbitSteps).getByText("Perturb")).toBeVisible();
    expect(within(orbitSteps).getByText("Evaluate")).toBeVisible();
    expect(within(orbitSteps).getByText("Update")).toBeVisible();
    expect(orbitFigure.querySelector(".orbit-update-flow")).toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-update-component")).not.toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-update-parallel")).not.toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-update-resultant")).toBeInTheDocument();
    expect(orbitFigure.querySelectorAll(".orbit-update-weighted-direction")).toHaveLength(10);
    expect(orbitFigure.querySelector(".orbit-update-contribution")).not.toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-update-summary")).toHaveTextContent("ALLSAMPLESWEIGHTEDSUM");
    expect(orbitFigure.querySelector(".orbit-update-summary")).not.toHaveTextContent("Σ");
    expect(orbitFigure).not.toHaveTextContent("w₁δ₁");
    expect(within(orbitSteps).getByText("Weight directions · move model")).toBeInTheDocument();
    expect(orbitFigure.querySelector(".orbit-next-core")).toBeInTheDocument();
    const longestWeightedDirection = orbitFigure.querySelector<SVGPolylineElement>('.orbit-update-weighted-direction[data-weight="5"]');
    expect(longestWeightedDirection).toBeInTheDocument();
    expect(longestWeightedDirection).toHaveAttribute("marker-mid", "url(#orbit-contribution-arrow)");
    const shortestWeightedDirection = orbitFigure.querySelector<SVGPolylineElement>('.orbit-update-weighted-direction[data-weight="1"]');
    expect(longestWeightedDirection).toHaveAttribute("data-weighted-length", "77");
    expect(shortestWeightedDirection).toHaveAttribute("data-weighted-length", "24");
    expect(Number(longestWeightedDirection?.dataset.weightedLength) - Number(shortestWeightedDirection?.dataset.weightedLength)).toBeGreaterThanOrEqual(50);
    const [, weightedEnd, samplePoint] = longestWeightedDirection?.getAttribute("points")?.split(" ").map((point) => point.split(",").map(Number)) ?? [];
    expect(Math.hypot((samplePoint?.[0] ?? 0) - (weightedEnd?.[0] ?? 0), (samplePoint?.[1] ?? 0) - (weightedEnd?.[1] ?? 0))).toBeGreaterThanOrEqual(12);
    const directionGradient = (direction: SVGPolylineElement | null) => {
      const selector = direction?.getAttribute("stroke")?.replace(/^url\((.*)\)$/, "$1");
      return selector ? orbitFigure.querySelector(`${selector} stop`) : null;
    };
    const highRewardGradient = directionGradient(longestWeightedDirection);
    const lowRewardGradient = directionGradient(shortestWeightedDirection);
    expect(highRewardGradient).toHaveAttribute("stop-color", "#1a6b4a");
    expect(lowRewardGradient).toHaveAttribute("stop-color", "#aab9a5");
    expect(screen.getByText("Model Scalability")).toBeInTheDocument();
    expect(screen.getByText("Flexibility")).toBeInTheDocument();
    expect(screen.getByText("Long-Horizon Scalability")).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Full-parameter ES optimization from scalar environment rewards." })).toBeInTheDocument();
    expect(screen.getByText("Agentic ESOpt samples parameter perturbations around the current LLM, evaluates the perturbed agents with scalar environment rewards, and applies a reward-weighted parameter update.")).toBeInTheDocument();
    expect(screen.getByText("ES enables full-parameter optimization with only minimal, inference-level GPU memory, making it possible to fine-tune large LLMs.")).toBeInTheDocument();
    expect(screen.getByText("Its lightweight, black-box feedback interface makes ES fine-tuning easy to compose with prompt-space evolution (e.g., skill optimization & test-time compute).")).toBeInTheDocument();
    expect(screen.getByText("ES performs trajectory-level parameter attribution without decomposing rewards across horizons, yielding better scalability than Agentic RL as the horizon length grows.")).toBeInTheDocument();
    expect(screen.queryByText("Model scalable")).not.toBeInTheDocument();
    expect(screen.queryByText("Environment flexible")).not.toBeInTheDocument();
    expect(screen.queryByText("Horizon robust")).not.toBeInTheDocument();
    expect(screen.getAllByRole("link", { name: "View code ↗" })[0]).toHaveAttribute("href", "https://github.com/zz1358m/Agentic-ESOpt");
    const scalingLink = screen.getByRole("link", { name: "Open the scaling matrix →" });
    expect(scalingLink).toHaveClass("scaling-callout__link");
    expect(scalingLink).toHaveAttribute("href", expect.stringMatching(/scaling\/$/));
    expect(screen.getByText("Five environments · result highlights")).toBeInTheDocument();
    expect(screen.queryByText("Five environments · selected results")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "Five environments, each with a matched comparison." })).toBeInTheDocument();
    const environmentComparisons = screen.getByRole("list", { name: "Five environment comparisons" });
    expect(within(environmentComparisons).getByText("40.63% → 53.13%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("+30.77%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("+21.44%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("48.0% → 52.5%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("+9.38%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("29.47% → 36.16%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("+22.70%")).toBeInTheDocument();
    expect(within(environmentComparisons).getByText("77.78%")).toBeInTheDocument();
    expect(screen.queryByText("+12.50 pts")).not.toBeInTheDocument();
    expect(screen.getByRole("heading", {
      level: 2,
      name: "Agentic ESOpt: Fine-Tuning Long-Horizon LLM Agents with Minimal GPU Memory Requirements",
    })).toBeInTheDocument();
    expect(screen.getByRole("heading", { level: 2, name: "How do backbone size and ES population change the result?" })).toBeInTheDocument();
    expect(screen.getByText("The experiment provides initial population-sensitivity evidence: doubling G changes final-test success by +677.0% for 4B but 0.0% for 9B. That indicates stronger backbones may need fewer sampled directions.")).toBeInTheDocument();
    expect(screen.getByText("Setup: 15-turn Sudoku; Qwen3.5-{4B, 9B}; G ∈ {8, 16}. G is the number of perturbation directions per ES update, not a physical compute-node count.")).toHaveClass("scaling-callout__note");
    expect(screen.getByText("Loading scaling results…")).toBeInTheDocument();
  });

  it("pauses the homepage ES animation when its figure leaves the viewport", () => {
    class MockIntersectionObserver {
      constructor(private callback: IntersectionObserverCallback) {}
      observe(target: Element) {
        this.callback([{ isIntersecting: false, target } as IntersectionObserverEntry], this as unknown as IntersectionObserver);
      }
      disconnect() {}
      unobserve() {}
      takeRecords() { return []; }
      readonly root = null;
      readonly rootMargin = "0px";
      readonly thresholds = [0];
    }
    vi.stubGlobal("IntersectionObserver", MockIntersectionObserver);

    render(<HomePage />);

    expect(screen.getByRole("figure", { name: "Interactive ES generation stages" })).toHaveClass("hero-orbit--paused");
  });
});
