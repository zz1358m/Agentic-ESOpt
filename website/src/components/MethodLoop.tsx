import { Tex } from "./Tex";

const steps = [
  { number: "01", title: "Current model", description: <>Start from the current LLM parameters <Tex math={"\\theta_t"} label="theta t"/>.</> },
  { number: "02", title: "Sample perturbations", description: "Sample G full-parameter perturbations around the current LLM." },
  { number: "03", title: "Evaluate agents", description: "Evaluate each perturbed agent through a complete environment trajectory." },
  { number: "04", title: "Collect rewards", description: "Obtain one scalar trajectory reward Rᵢ for each perturbed agent." },
  { number: "05", title: "Normalize rewards", description: "Normalize rewards within the population using a z-score." },
  { number: "06", title: "Reward-weighted update", description: "Weight each full-parameter perturbation by its normalized reward R̂ᵢ." },
  { number: "07", title: "Next generation", description: <>Apply the update to obtain <Tex math={"\\theta_{t+1}"} label="theta t plus 1"/>.</> },
] as const;

const equations = [
  {
    number: "01",
    title: "Perturb",
    equation: "\\theta_i = \\theta_t + \\sigma \\epsilon_i",
    label: "theta i equals theta t plus sigma epsilon i",
    detail: "Sample G full-parameter directions around the current model.",
  },
  {
    number: "02",
    title: "Evaluate",
    equation: "R_i = R(\\tau_i)",
    label: "reward i equals reward of trajectory i",
    detail: "Run the complete agent trajectory and retain one scalar reward.",
  },
  {
    number: "03",
    title: "Update",
    equation: "\\theta_{t+1} = \\theta_t + \\frac{\\alpha}{G} \\sum_{i=1}^{G} \\hat{R}_i \\epsilon_i",
    label: "theta t plus 1 equals theta t plus alpha over G times the sum of normalized reward i times epsilon i",
    detail: "Z-score rewards, weight every sampled direction, and update once.",
  },
] as const;

export function MethodLoop() {
  return (
    <div className="method-lab">
      <div className="method-mechanism" role="group" aria-label="Agentic ESOpt mechanism">
        <span className="method-mechanism__eyebrow">One ES generation</span>
        <div className="method-equations">
          {equations.map((item) => (
            <article key={item.number}>
              <header><span>{item.number}</span><strong>{item.title}</strong></header>
              <p className="method-equation"><Tex math={item.equation} label={item.label}/></p>
              <p>{item.detail}</p>
            </article>
          ))}
        </div>
        <p className="method-mechanism__note"><strong>Gradient-free:</strong> scalar rewards drive the update without differentiating through the agent–environment interaction.</p>
      </div>
      <div className="method-controls">
        <ol className="method-steps" aria-label="Agentic ESOpt optimization steps">
          {steps.map((step) => (
            <li key={step.number}>
              <span>{step.number}</span>
              <div><strong>{step.title}</strong><small>{step.description}</small></div>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
