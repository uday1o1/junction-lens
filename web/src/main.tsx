import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import "./styles.css";

const root = document.querySelector<HTMLDivElement>("#root");
if (root === null) throw new Error("JunctionLens application root is missing");
createRoot(root).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
