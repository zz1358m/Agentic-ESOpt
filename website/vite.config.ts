import { resolve } from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

const pages = [
  "index.html",
  "tasks/sudoku/index.html",
  "tasks/math/index.html",
  "tasks/docvqa/index.html",
  "tasks/webarena/index.html",
  "tasks/ahd/index.html",
  "scaling/index.html",
  "paper/index.html",
];

export default defineConfig({
  base: "./",
  plugins: [react()],
  build: {
    rollupOptions: {
      input: Object.fromEntries(
        pages.map((page) => [page.replace(/\/index\.html$/, "") || "home", resolve(__dirname, page)]),
      ),
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./tests/setup.ts",
  },
});
