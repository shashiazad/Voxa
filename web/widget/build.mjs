import { build } from "esbuild";
await build({
  entryPoints: ["src/widget.ts"],
  bundle: true, minify: true, format: "iife", target: "es2020", sourcemap: false,
  outfile: "../dashboard/public/widget.js",
  legalComments: "none",
});
console.log("widget.js built");
