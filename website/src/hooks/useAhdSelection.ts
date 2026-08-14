import { useUrlNumber } from "./useUrlNumber";
import { useUrlString } from "./useUrlString";

export type AhdConfigSelection = {
  problem: string;
  mode: string;
  outerMethod: string;
  agenticESOpt: boolean;
  budget: number;
  repeat: number;
};

export function useAhdSelection() {
  const [problem, setProblem] = useUrlString("problem", "TSP");
  const [mode, setMode] = useUrlString("mode", "ACO");
  const [outerMethod, setOuterMethod] = useUrlString("outer", "Sample");
  const [agentic, setAgentic] = useUrlString("agentic", "on");
  const [budget, setBudget] = useUrlNumber("budget", 1000);
  const [repeat, setRepeat] = useUrlNumber("repeat", 1);
  const [generation, setGeneration] = useUrlNumber("generation", 50);

  const selection: AhdConfigSelection = { problem, mode, outerMethod, agenticESOpt: agentic === "on", budget, repeat };
  const apply = (config: AhdConfigSelection, nextGeneration: number) => {
    setProblem(config.problem);
    setMode(config.mode);
    setOuterMethod(config.outerMethod);
    setAgentic(config.agenticESOpt ? "on" : "off");
    setBudget(config.budget);
    setRepeat(config.repeat);
    setGeneration(nextGeneration);
  };
  return { selection, generation, setGeneration, apply };
}
