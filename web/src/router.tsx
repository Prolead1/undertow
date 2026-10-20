import { lazy, Suspense, type ReactNode } from "react";

import App from "./App";
import { Navigate, useLocation } from "./router/primitives";

/**
 * App routes (CONTRACTS §8). Chart-heavy views are `React.lazy` so the Plotly
 * bundle stays out of the initial chunk; `App` (the shell) is eager.
 */
const LiveCockpit = lazy(() => import("./views/LiveCockpit"));
const PolicyReplay = lazy(() => import("./views/PolicyReplay"));
const Leaderboard = lazy(() => import("./views/Leaderboard"));
const Compare = lazy(() => import("./views/Compare"));

const ROUTE_FALLBACK = (
  <div
    role="status"
    aria-live="polite"
    className="rounded-control border border-dashed border-border bg-bg-base/40 p-6 text-center text-xs text-text-muted"
  >
    Loading view…
  </div>
);

const ROUTES: Record<string, ReactNode> = {
  "/live": <LiveCockpit />,
  "/replay": <PolicyReplay />,
  "/leaderboard": <Leaderboard />,
  "/compare": <Compare />,
};

/** The app shell plus the view for the current path; unknown paths go to `/live`. */
export function AppRouter() {
  const { pathname, search } = useLocation();
  const view = ROUTES[pathname];
  if (view === undefined) {
    return <Navigate to={`/live${search}`} replace />;
  }
  return (
    <App>
      <Suspense fallback={ROUTE_FALLBACK}>{view}</Suspense>
    </App>
  );
}

export {
  BrowserRouter,
  Link,
  MemoryRouter,
  Navigate,
  useLocation,
  useNavigate,
  useSearchParams,
} from "./router/primitives";
export type { LinkProps, Location, NavigateOptions, To } from "./router/primitives";
