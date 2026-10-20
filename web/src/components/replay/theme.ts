/** CONTRACTS §9 tokens (Plotly does not resolve CSS variables). */
export const TOKEN = {
  accent: "#22D3EE",
  accent2: "#A78BFA",
  textPrimary: "#E6EAF2",
  textMuted: "#8B93A7",
  bull: "#34D399",
  bear: "#F87171",
  sideways: "#94A3B8",
  rebalance: "#38BDF8",
  highVol: "#FBBF24",
  bgBase: "#0B0E14",
  grid: "#1A2130",
} as const;

/**
 * Translate a `#RRGGBB` token into an `rgba()` string at the given alpha.
 *
 * Plotly does not resolve CSS variables, so translucent fills have to be
 * spelt out; deriving them from the same §9 tokens as their solid strokes
 * keeps both in lockstep.
 */
export function withAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const red = Number.parseInt(value.slice(0, 2), 16);
  const green = Number.parseInt(value.slice(2, 4), 16);
  const blue = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

/**
 * Inlined dark theme for client-built Plotly specs.
 *
 * `plotly.js-dist-min` ignores unregistered template names, so the browser has
 * no `undertow_dark` template to inherit from; the §9 colours are listed here
 * instead.
 */
export const DARK_LAYOUT = {
  paper_bgcolor: "#11161F",
  plot_bgcolor: "#11161F",
  font: { color: "#E6EAF2", family: "Inter, system-ui, sans-serif", size: 12 },
} as const;
