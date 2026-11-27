// Saves positions and clusters into a graph JSON file: node dist/bake.mjs <file>
import { readFileSync, writeFileSync } from "node:fs";
import { bake } from "./layout";

const file = process.argv[2];
if (!file) throw new Error("usage: bake <graph.json>");
const data = JSON.parse(readFileSync(file, "utf8"));
writeFileSync(file, JSON.stringify(bake(data)));
