import React from "react";
import ReactDOM from "react-dom/client";

import "./theme/tokens.css";
import { AppRouter, BrowserRouter } from "./router";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root element");

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <BrowserRouter>
      <AppRouter />
    </BrowserRouter>
  </React.StrictMode>,
);
