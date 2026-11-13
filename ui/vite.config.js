import { fileURLToPath } from "node:url"

import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"
import { viteSingleFile } from "vite-plugin-singlefile"

// One self-contained index.html (script, styles and the noise texture inlined), written where
// the Python app reads it. A single file needs no file:// access from inside the web view.
export default defineConfig({
  plugins: [tailwindcss(), viteSingleFile()],
  esbuild: { jsx: "automatic" },
  // "@/..." is how shadcn/ui components import shared code; keep it so they copy in unchanged.
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  build: { outDir: "../src/macecho/ui", emptyOutDir: true },
})
