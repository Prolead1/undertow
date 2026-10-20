import { Navigate, useLocation } from "react-router-dom";

/** `/` and unknown paths redirect to `/live` while keeping the selection query. */
export function RedirectToLive() {
  const location = useLocation();
  return <Navigate to={`/live${location.search}`} replace />;
}
