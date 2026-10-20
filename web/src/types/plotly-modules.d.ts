// `plotly.js-dist-min` and `react-plotly.js/factory` ship no type declarations.
// The factory is typed from the upstream `plotly.js` / `react-plotly.js` types.
declare module "plotly.js-dist-min" {
  import type * as Plotly from "plotly.js";
  const plotly: typeof Plotly;
  export default plotly;
}

declare module "react-plotly.js/factory" {
  import type { PlotParams } from "react-plotly.js";
  import type * as Plotly from "plotly.js";
  const createPlotlyComponent: (plotly: typeof Plotly) => React.ComponentType<PlotParams>;
  export default createPlotlyComponent;
}
