import { StrictMode, type ReactNode } from "react";

import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import {
  AppRouter,
  BrowserRouter,
  Link,
  MemoryRouter,
  Navigate,
  useLocation,
  useNavigate,
  useSearchParams,
} from "../router";

// The shell and the chart-heavy views are irrelevant here; the router is under
// test, so `App` (and the lazily-loaded view) are stubbed.
vi.mock("../App", () => ({
  default: ({ children }: { children: ReactNode }) => (
    <div>
      <span data-testid="shell">shell</span>
      {children}
    </div>
  ),
}));
vi.mock("../views/LiveCockpit", () => ({ default: () => <div>live-view</div> }));

function LocationProbe() {
  const { pathname, search } = useLocation();
  return <span data-testid="loc">{`${pathname}${search}`}</span>;
}

beforeEach(() => {
  window.history.replaceState(null, "", "/live");
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("router primitives", () => {
  it("MemoryRouter exposes the initial entry and Link navigates", () => {
    render(
      <MemoryRouter initialEntries={["/live?run_id=r1"]}>
        <Link to="/replay">Replay</Link>
        <LocationProbe />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("loc").textContent).toBe("/live?run_id=r1");

    fireEvent.click(screen.getByText("Replay"));
    expect(screen.getByTestId("loc").textContent).toBe("/replay");
    // Navigating with an absolute path drops the previous query.
    expect(screen.getByText("Replay").getAttribute("href")).toBe("/replay");
  });

  it("useNavigate accepts a string and an object destination", () => {
    function NavProbe() {
      const navigate = useNavigate();
      return (
        <div>
          <button onClick={() => navigate("/replay?run_id=r2")}>string</button>
          <button onClick={() => navigate({ pathname: "/compare", search: "run_ids=a,b" })}>
            object
          </button>
        </div>
      );
    }
    render(
      <MemoryRouter initialEntries={["/live"]}>
        <NavProbe />
        <LocationProbe />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByText("string"));
    expect(screen.getByTestId("loc").textContent).toBe("/replay?run_id=r2");

    fireEvent.click(screen.getByText("object"));
    expect(screen.getByTestId("loc").textContent).toBe("/compare?run_ids=a,b");
  });

  it("useSearchParams replaces the query string", () => {
    function ParamsProbe() {
      const [params, setParams] = useSearchParams();
      return (
        <button
          onClick={() => {
            const next = new URLSearchParams(params);
            next.set("seed", "9");
            setParams(next);
          }}
        >
          set
        </button>
      );
    }
    render(
      <MemoryRouter initialEntries={["/replay?run_id=r"]}>
        <ParamsProbe />
        <LocationProbe />
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByText("set"));
    expect(screen.getByTestId("loc").textContent).toBe("/replay?run_id=r&seed=9");
  });

  it("useSearchParams honours { replace } in the browser router", () => {
    function ReplaceProbe() {
      const [, setParams] = useSearchParams();
      return <button onClick={() => setParams(new URLSearchParams("run_id=z"), { replace: true })}>r</button>;
    }
    const replaceState = vi.spyOn(window.history, "replaceState");
    const pushState = vi.spyOn(window.history, "pushState");
    render(
      <BrowserRouter>
        <ReplaceProbe />
      </BrowserRouter>,
    );

    fireEvent.click(screen.getByText("r"));
    expect(replaceState).toHaveBeenCalledTimes(1);
    expect(pushState).not.toHaveBeenCalled();
  });

  it("Link honours modifier keys and middle clicks", () => {
    render(
      <MemoryRouter initialEntries={["/live"]}>
        <Link to="/replay">Replay</Link>
        <LocationProbe />
      </MemoryRouter>,
    );
    fireEvent.click(screen.getByText("Replay"), { metaKey: true });
    expect(screen.getByTestId("loc").textContent).toBe("/live");

    fireEvent.click(screen.getByText("Replay"), { button: 1 });
    expect(screen.getByTestId("loc").textContent).toBe("/live");
  });

  it("Navigate redirects on mount", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <Navigate to="/live?run_id=r3" replace />
        <LocationProbe />
      </MemoryRouter>,
    );
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toBe("/live?run_id=r3"));
  });

  it("BrowserRouter writes history and reacts to popstate", () => {
    render(
      <BrowserRouter>
        <Link to="/replay">Replay</Link>
        <LocationProbe />
      </BrowserRouter>,
    );

    fireEvent.click(screen.getByText("Replay"));
    expect(window.location.pathname).toBe("/replay");
    expect(screen.getByTestId("loc").textContent).toBe("/replay");

    act(() => {
      window.history.pushState(null, "", "/leaderboard?run_id=r4");
      window.dispatchEvent(new PopStateEvent("popstate"));
    });
    expect(screen.getByTestId("loc").textContent).toBe("/leaderboard?run_id=r4");
  });

  it("pushes exactly one history entry per navigation under StrictMode", () => {
    const pushState = vi.spyOn(window.history, "pushState");
    render(
      <StrictMode>
        <BrowserRouter>
          <Link to="/replay">Replay</Link>
        </BrowserRouter>
      </StrictMode>,
    );
    fireEvent.click(screen.getByText("Replay"));
    expect(pushState).toHaveBeenCalledTimes(1);
  });
});

describe("AppRouter", () => {
  it("redirects / to /live preserving the query", async () => {
    render(
      <MemoryRouter initialEntries={["/?run_id=r1"]}>
        <AppRouter />
        <LocationProbe />
      </MemoryRouter>,
    );
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toBe("/live?run_id=r1"));
  });

  it("redirects unknown paths to /live", async () => {
    render(
      <MemoryRouter initialEntries={["/nope?seed=2"]}>
        <AppRouter />
        <LocationProbe />
      </MemoryRouter>,
    );
    await waitFor(() => expect(screen.getByTestId("loc").textContent).toBe("/live?seed=2"));
  });

  it("renders a known route inside the shell", async () => {
    render(
      <MemoryRouter initialEntries={["/live"]}>
        <AppRouter />
      </MemoryRouter>,
    );
    expect(screen.getByTestId("shell")).toBeTruthy();
    expect(await screen.findByText("live-view")).toBeTruthy();
  });
});
