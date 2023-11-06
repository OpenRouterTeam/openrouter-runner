import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main() {
  const prompt = 'What was Project A119 and what were its objectives?';

  await completion(prompt);
}

runIfCalledAsScript(main, import.meta.url);
