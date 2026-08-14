import { useEffect, useState } from "react";

type Props = {
  generations: number[];
  value: number;
  onChange: (generation: number) => void;
  label?: string;
};

export function GenerationControl({ generations, value, onChange, label = "ES generation" }: Props) {
  const [playing, setPlaying] = useState(false);
  const index = Math.max(0, generations.indexOf(value));
  const visibleTicks = generations.length <= 6
    ? generations
    : Array.from({ length: 6 }, (_, tick) => generations[Math.round((tick * (generations.length - 1)) / 5)]);

  useEffect(() => {
    if (!playing || generations.length < 2) return;
    const timer = window.setInterval(() => {
      const next = (generations.indexOf(value) + 1) % generations.length;
      onChange(generations[next]);
      if (next === generations.length - 1) setPlaying(false);
    }, 850);
    return () => window.clearInterval(timer);
  }, [generations, onChange, playing, value]);

  const move = (direction: -1 | 1) => {
    const nextIndex = Math.min(generations.length - 1, Math.max(0, index + direction));
    onChange(generations[nextIndex]);
  };

  return (
    <div className="generation-control" aria-label={`${label} control`}>
      <div className="generation-control__meta">
        <span>{label}</span>
        <strong>{value < 0 ? "Base" : value}</strong>
      </div>
      <div className="generation-control__row">
        <button className="icon-button" type="button" aria-label="Previous generation" onClick={() => move(-1)} disabled={index === 0}>←</button>
        <button className="icon-button" type="button" aria-label={playing ? "Pause generations" : "Play generations"} onClick={() => setPlaying((current) => !current)}>{playing ? "Ⅱ" : "▶"}</button>
        <input
          aria-label={label}
          type="range"
          min={0}
          max={Math.max(0, generations.length - 1)}
          step={1}
          value={index}
          onChange={(event) => onChange(generations[Number(event.target.value)])}
        />
        <button className="icon-button" type="button" aria-label="Next generation" onClick={() => move(1)} disabled={index === generations.length - 1}>→</button>
      </div>
      <div className="generation-control__ticks" aria-hidden="true">
        {visibleTicks.map((generation) => <span key={generation}>{generation < 0 ? "Base" : generation}</span>)}
      </div>
    </div>
  );
}
