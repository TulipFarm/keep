export const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");

export function sh(line: string): string {
  let first = true;
  return line
    .split(/(\s+)/)
    .map((t) => {
      if (!t.trim()) return t;
      const e = esc(t);
      if (first) { first = false; return `<span class="t-cmd">${e}</span>`; }
      if (t.startsWith("--") || /^-\w$/.test(t)) return `<span class="t-flag">${e}</span>`;
      if (/^(&|&&|\||;)$/.test(t)) return `<span class="t-op">${e}</span>`;
      if (/^https?:\/\//.test(t) || t.includes("/") || t.includes(".")) return `<span class="t-str">${e}</span>`;
      return `<span class="t-arg">${e}</span>`;
    })
    .join("");
}
