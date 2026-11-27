import { fileURLToPath } from "node:url";
import { defineConfig } from "astro/config";

const devUi = fileURLToPath(new URL("../crates/keep-server/dev-ui/src", import.meta.url));

export default defineConfig({
  site: "https://keep.tulipfarm.ai",
  vite: {
    // The graph demo imports the dev UI's viewer; resolve its libraries
    // from this package and let the dev server read its source.
    resolve: { dedupe: ["graphology", "graphology-communities-louvain", "graphology-layout-forceatlas2", "sigma"] },
    server: { fs: { allow: [".", devUi] } },
  },
});
