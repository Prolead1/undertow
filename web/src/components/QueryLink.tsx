import type { ReactNode } from "react";
import { Link, useLocation, type LinkProps } from "../router";

import { withPreservedSearch } from "../navigation";

export interface QueryLinkProps extends Omit<LinkProps, "to"> {
  to: string;
  children: ReactNode;
}

/** Navigation link that preserves the current selection query params. */
export function QueryLink({ to, children, ...rest }: QueryLinkProps) {
  const location = useLocation();
  return (
    <Link to={withPreservedSearch(to, location.search)} {...rest}>
      {children}
    </Link>
  );
}
