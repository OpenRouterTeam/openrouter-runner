/**
 * @type {import('prettier').Options}
 */
export default {
  arrowParens: "always",
  singleQuote: true,
  tabWidth: 2,
  trailingComma: "none",
  plugins: ["@ianvs/prettier-plugin-sort-imports"],
  importOrder: [
    "<BUILTIN_MODULES>", // Node.js built-in modules
    "<THIRD_PARTY_MODULES>", // Imports not matched by other special words or groups.
    "",
    "^@/(.*)$",
    "",
    "^[./]",
  ],
};
