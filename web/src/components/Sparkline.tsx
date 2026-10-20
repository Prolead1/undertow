import { cn } from "./cn";

export interface SparklineProps {
  values: readonly number[];
  width?: number;
  height?: number;
  label?: string;
  tone?: "accent" | "bull" | "bear" | "accent-2";
  className?: string;
}

const STROKE: Record<NonNullable<SparklineProps["tone"]>, string> = {
  accent: "var(--accent)",
  bull: "var(--bull)",
  bear: "var(--bear)",
  "accent-2": "var(--accent-2)",
};

/** Dependency-free sparkline for run cards and stats (charts use PlotlyFigure). */
export function Sparkline({
  values,
  width = 120,
  height = 28,
  label,
  tone = "accent",
  className,
}: SparklineProps) {
  const finite = values.filter((value) => Number.isFinite(value));
  const points =
    finite.length === 0
      ? ""
      : finite
          .map((value, index) => {
            const min = Math.min(...finite);
            const max = Math.max(...finite);
            const range = max - min || 1;
            const x = finite.length === 1 ? width / 2 : (index / (finite.length - 1)) * width;
            const y = height - ((value - min) / range) * (height - 2) - 1;
            return `${x.toFixed(2)},${y.toFixed(2)}`;
          })
          .join(" ");

  return (
    <svg
      className={cn("shrink-0", className)}
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      focusable="false"
    >
      {points ? (
        <polyline
          points={points}
          fill="none"
          stroke={STROKE[tone]}
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      ) : (
        <line
          x1={0}
          y1={height / 2}
          x2={width}
          y2={height / 2}
          stroke="var(--grid)"
          strokeWidth={1}
        />
      )}
    </svg>
  );
}
