import { extent, line, max, min, scaleLinear } from "d3";
import type { CurveSeries } from "../types";

type Props = {
  series: CurveSeries[];
  generation: number;
  title: string;
  valueFormatter?: (value: number) => string;
  lowerIsBetter?: boolean;
  xLabel?: string;
};

const WIDTH = 760;
const HEIGHT = 330;
const MARGIN = { top: 28, right: 24, bottom: 48, left: 58 };

export function MetricChart({ series, generation, title, valueFormatter = (value) => value.toFixed(2), lowerIsBetter = false, xLabel = "ES generation" }: Props) {
  const all = series.flatMap((item) => item.points);
  if (!all.length) return <div className="empty-state">Curve data not recorded.</div>;
  const [xMin = 0, xMax = 1] = extent(all, (point) => point.generation);
  const yMin = min(all, (point) => point.value) ?? 0;
  const yMax = max(all, (point) => point.value) ?? 1;
  const pad = Math.max((yMax - yMin) * 0.15, 0.03);
  const x = scaleLinear().domain([xMin, xMax || 1]).range([MARGIN.left, WIDTH - MARGIN.right]);
  const y = scaleLinear().domain([Math.max(0, yMin - pad), yMax + pad]).nice().range([HEIGHT - MARGIN.bottom, MARGIN.top]);
  const path = line<{ generation: number; value: number }>()
    .x((point) => x(point.generation))
    .y((point) => y(point.value));
  const cursorX = x(Math.min(xMax, Math.max(xMin, generation)));
  const selectedXLabel = xLabel === "ES generation" ? "generation" : xLabel.toLowerCase();
  const xTicks = x.ticks(5);
  const yTicks = y.ticks(4);
  const selectedPoints = series.filter((item) => item.points.length).map((item) => {
    const point = item.points.reduce((nearest, candidate) => (
      Math.abs(candidate.generation - generation) < Math.abs(nearest.generation - generation) ? candidate : nearest
    ), item.points[0]);
    const initial = item.points[0];
    return { item, point, delta: point.value - initial.value };
  });
  const selectedSummary = selectedPoints.map(({ item, point, delta }) => (
    `${item.label}: ${valueFormatter(point.value)} at ${selectedXLabel} ${point.generation}, ${delta >= 0 ? "+" : ""}${valueFormatter(delta)} from initial`
  )).join(". ");

  return (
    <figure className="chart-card">
      <figcaption>
        <div>
          <span className="eyebrow">Observed performance</span>
          <h3>{title}</h3>
        </div>
        <span className="direction-note">{lowerIsBetter ? "↓ lower is better" : "↑ higher is better"}</span>
      </figcaption>
      <svg className="metric-chart" viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`${title}. Selected ${selectedXLabel} ${generation}`}>
        <desc>{selectedSummary}</desc>
        <g className="grid-lines">
          {yTicks.map((tick) => <line key={tick} x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(tick)} y2={y(tick)} />)}
        </g>
        <g className="axis-labels">
          {yTicks.map((tick) => <text key={tick} x={MARGIN.left - 12} y={y(tick) + 4} textAnchor="end">{valueFormatter(tick)}</text>)}
          {xTicks.map((tick) => <text key={tick} x={x(tick)} y={HEIGHT - 18} textAnchor="middle">{tick}</text>)}
          <text x={WIDTH - MARGIN.right} y={HEIGHT - 3} textAnchor="end">{xLabel}</text>
        </g>
        {series.map((item, seriesIndex) => (
          <g key={item.id} className={`series series--${item.kind}`}>
            <path d={path(item.points) ?? undefined} className={`series-path series-color-${seriesIndex % 5}`} />
            {item.points.map((point) => <circle key={`${point.generation}-${point.value}`} cx={x(point.generation)} cy={y(point.value)} r={item.kind === "eval" ? 4.2 : 2.2} className={`series-dot series-color-${seriesIndex % 5}`} />)}
          </g>
        ))}
        <g aria-label={`Selected ${selectedXLabel} ${generation}`}>
          <line className="cursor-line" x1={cursorX} x2={cursorX} y1={MARGIN.top} y2={HEIGHT - MARGIN.bottom} />
          <circle className="cursor-handle" cx={cursorX} cy={MARGIN.top + 2} r={6} />
        </g>
      </svg>
      <div className="chart-legend">
        {series.map((item, index) => <span key={item.id}><i className={`legend-dot series-color-${index % 5}`} />{item.label}</span>)}
      </div>
      <div className="chart-readings" aria-label="Current metrics">
        {selectedPoints.map(({ item, point, delta }, index) => <div key={item.id}>
          <i className={`legend-dot series-color-${index % 5}`} />
          <span>{item.label} · {selectedXLabel} {point.generation}</span>
          <strong>{valueFormatter(point.value)}</strong>
          <small>{delta >= 0 ? "+" : ""}{valueFormatter(delta)} from initial</small>
        </div>)}
      </div>
    </figure>
  );
}
