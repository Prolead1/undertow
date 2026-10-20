// `plotly.js-dist-min` ships no type declarations; it has the same runtime API
// as `plotly.js`, whose `@types` are used here.
declare module "plotly.js-dist-min" {
  import type * as Plotly from "plotly.js";
  const plotly: typeof Plotly;
  export default plotly;
}
