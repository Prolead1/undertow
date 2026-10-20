import type { Config } from "tailwindcss";

/**
 * Tailwind v4 config.  The semantic token names mirror CONTRACTS.md §9 and
 * resolve to the CSS variables declared in `src/theme/tokens.css`, so the
 * Python Plotly theme and the React UI can never drift.
 */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        "bg-base": "var(--bg-base)",
        "bg-surface": "var(--bg-surface)",
        "bg-elevated": "var(--bg-elevated)",
        border: "var(--border)",
        "text-primary": "var(--text-primary)",
        "text-muted": "var(--text-muted)",
        grid: "var(--grid)",
        accent: "var(--accent)",
        "accent-2": "var(--accent-2)",
        bull: "var(--bull)",
        bear: "var(--bear)",
        sideways: "var(--sideways)",
        "high-vol": "var(--high-vol)",
        hold: "var(--hold)",
        rebalance: "var(--rebalance)",
      },
      borderRadius: {
        panel: "8px",
        control: "6px",
      },
    },
  },
} satisfies Config;
