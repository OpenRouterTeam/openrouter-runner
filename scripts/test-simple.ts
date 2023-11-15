import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main() {
  const prompt = `USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:`;

  await completion(prompt);
}

runIfCalledAsScript(main, import.meta.url);
