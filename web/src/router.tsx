import { createBrowserRouter, type RouteObject } from "react-router-dom";

import App from "./App";
import { RedirectToLive } from "./components/RedirectToLive";
import Compare from "./views/Compare";
import Leaderboard from "./views/Leaderboard";
import LiveCockpit from "./views/LiveCockpit";
import PolicyReplay from "./views/PolicyReplay";

/** CONTRACTS §8 route table; every route lives inside the shared App shell. */
export const appRoutes: RouteObject[] = [
  {
    path: "/",
    element: <App />,
    children: [
      { index: true, element: <RedirectToLive /> },
      { path: "live", element: <LiveCockpit /> },
      { path: "replay", element: <PolicyReplay /> },
      { path: "leaderboard", element: <Leaderboard /> },
      { path: "compare", element: <Compare /> },
      { path: "*", element: <RedirectToLive /> },
    ],
  },
];

export function createAppRouter() {
  return createBrowserRouter(appRoutes, {
    future: { v7_relativeSplatPath: true },
  });
}
