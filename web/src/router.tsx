import { lazy, Suspense, type ReactNode } from "react";
import { createBrowserRouter, type RouteObject } from "react-router-dom";

import App from "./App";
import { RedirectToLive } from "./components/RedirectToLive";

/**
 * Chart-heavy views are route-split (`React.lazy`) so the ~5 MB Plotly bundle
 * is not part of the initial chunk. `App`, the run rail and the redirect are
 * eager because they are tiny and always visible.
 */
const LiveCockpit = lazy(() => import("./views/LiveCockpit"));
const PolicyReplay = lazy(() => import("./views/PolicyReplay"));
const Leaderboard = lazy(() => import("./views/Leaderboard"));
const Compare = lazy(() => import("./views/Compare"));

/** Shared Suspense fallback; kept as an element so this file exports no component. */
const ROUTE_FALLBACK = (
  <div
    role="status"
    aria-live="polite"
    className="rounded-control border border-dashed border-border bg-bg-base/40 p-6 text-center text-xs text-text-muted"
  >
    Loading view…
  </div>
);

/** Wrap a lazy route element in the shared Suspense boundary. */
function deferred(element: ReactNode) {
  return <Suspense fallback={ROUTE_FALLBACK}>{element}</Suspense>;
}

/** CONTRACTS §8 route table; every route lives inside the shared App shell. */
export const appRoutes: RouteObject[] = [
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <RedirectToLive /> },
      { path: "live", element: deferred(<LiveCockpit />) },
      { path: "replay", element: deferred(<PolicyReplay />) },
      { path: "leaderboard", element: deferred(<Leaderboard />) },
      { path: "compare", element: deferred(<Compare />) },
      { path: "*", element: <RedirectToLive /> },
    ],
  },
];

export function createAppRouter() {
  return createBrowserRouter(appRoutes, {
    future: { v7_relativeSplatPath: true },
  });
}
