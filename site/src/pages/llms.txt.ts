import { repo, docs } from "../data/site";

export const GET = () =>
  new Response(
    `# Keep

> Keep is open-source memory and knowledge for AI agents. It gives coding agents one shared memory per project over MCP. When a fact changes, Keep marks the old one as replaced and does not erase it. Apache-2.0. Research preview.

Status: Keep is a research preview. No benchmark results are published yet.

- [Homepage in Markdown](https://keep.tulipfarm.ai/index.md): what Keep does, where it runs, status, FAQ
- [Install instructions for agents](https://keep.tulipfarm.ai/install.md): the file a coding agent follows to install Keep
- [Memory](${docs.memory})
- [Knowledge](${docs.knowledge})
- [Server and agent setup](${docs.server})
- [Benchmark methodology](${docs.method})
- [Source code](${repo})
`,
    { headers: { "Content-Type": "text/plain; charset=utf-8" } },
  );
