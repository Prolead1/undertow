import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type AnchorHTMLAttributes,
  type ReactNode,
} from "react";

/**
 * Minimal client-side router: history + query string only, no dependencies.
 *
 * The dashboard has four static routes and uses the query string for selection
 * (`?run_id=&seed=&episode=`), so react-router's data APIs (loaders, actions,
 * nested route data) are unnecessary. This module covers exactly what the app
 * and its tests use: location, navigation, `Link`, `Navigate`, search params,
 * and an in-memory router for tests.
 */

export interface Location {
  pathname: string;
  search: string;
}

export type To = string | { pathname: string; search?: string };

export interface NavigateOptions {
  replace?: boolean;
}

interface RouterValue {
  location: Location;
  navigate: (to: To, options?: NavigateOptions) => void;
}

const RouterContext = createContext<RouterValue | null>(null);

const ROOT: Location = { pathname: "/", search: "" };

function splitTo(to: To, current: Location): Location {
  if (typeof to !== "string") {
    const search = to.search ?? "";
    return {
      pathname: to.pathname,
      search: search === "" || search.startsWith("?") ? search : `?${search}`,
    };
  }
  const [pathname = "", search = ""] = to.split("?", 2);
  return {
    pathname: pathname === "" ? current.pathname : pathname,
    search: search === "" ? "" : `?${search}`,
  };
}

/** Resolve a destination to an `href` string (used by `Link`). */
export function hrefOf(to: To): string {
  const { pathname, search } = splitTo(to, ROOT);
  return `${pathname}${search}`;
}

function useRouter(): RouterValue {
  const value = useContext(RouterContext);
  if (value === null) {
    throw new Error("router hooks must be used inside <BrowserRouter> or <MemoryRouter>");
  }
  return value;
}

function browserLocation(): Location {
  return { pathname: window.location.pathname, search: window.location.search };
}

export function BrowserRouter({ children }: { children: ReactNode }) {
  const [location, setLocation] = useState<Location>(browserLocation);

  useEffect(() => {
    const onPopState = (): void => setLocation(browserLocation());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((to: To, options?: NavigateOptions) => {
    const next = splitTo(to, browserLocation());
    const href = `${next.pathname}${next.search}`;
    if (options?.replace === true) window.history.replaceState(null, "", href);
    else window.history.pushState(null, "", href);
    setLocation(next);
  }, []);

  const value = useMemo<RouterValue>(() => ({ location, navigate }), [location, navigate]);
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function MemoryRouter({
  initialEntries = ["/"],
  children,
}: {
  initialEntries?: string[];
  children: ReactNode;
}) {
  const [location, setLocation] = useState<Location>(() =>
    splitTo(initialEntries[0] ?? "/", ROOT),
  );
  const navigate = useCallback((to: To) => {
    setLocation((current) => splitTo(to, current));
  }, []);
  const value = useMemo<RouterValue>(() => ({ location, navigate }), [location, navigate]);
  return <RouterContext.Provider value={value}>{children}</RouterContext.Provider>;
}

export function useLocation(): Location {
  return useRouter().location;
}

export function useNavigate(): RouterValue["navigate"] {
  return useRouter().navigate;
}

export function useSearchParams(): [
  URLSearchParams,
  (next: URLSearchParams, options?: NavigateOptions) => void,
] {
  const { location, navigate } = useRouter();
  const params = useMemo(() => new URLSearchParams(location.search), [location.search]);
  const setParams = useCallback(
    (next: URLSearchParams, options?: NavigateOptions) => {
      navigate({ pathname: location.pathname, search: next.toString() }, options);
    },
    [location.pathname, navigate],
  );
  return [params, setParams];
}

export interface LinkProps extends Omit<AnchorHTMLAttributes<HTMLAnchorElement>, "href"> {
  to: To;
  replace?: boolean;
  children?: ReactNode;
}

export function Link({ to, replace, onClick, children, ...rest }: LinkProps) {
  const { navigate } = useRouter();
  return (
    <a
      href={hrefOf(to)}
      onClick={(event) => {
        onClick?.(event);
        if (
          event.defaultPrevented ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey
        ) {
          return;
        }
        event.preventDefault();
        navigate(to, { replace });
      }}
      {...rest}
    >
      {children}
    </a>
  );
}

/** Declarative redirect: navigates on mount (and when the destination changes). */
export function Navigate({ to, replace }: { to: To; replace?: boolean }) {
  const navigate = useNavigate();
  // Depend on the resolved href (a string) rather than the `to` object, so a
  // fresh object literal does not re-navigate on every render.
  const href = hrefOf(to);
  useEffect(() => {
    navigate(href, { replace });
  }, [navigate, href, replace]);
  return null;
}
