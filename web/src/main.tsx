import React from "react";
import ReactDOM from "react-dom/client";
import { RouterProvider } from "react-router-dom";

import "./theme/tokens.css";
import { createAppRouter } from "./router";

const root = document.getElementById("root");
if (!root) throw new Error("missing #root element");

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <RouterProvider router={createAppRouter()} />
  </React.StrictMode>,
);
