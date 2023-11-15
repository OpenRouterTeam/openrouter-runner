import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main() {
  const prompt = `USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:`;

  await completion(
    prompt,
    {
      max_tokens: 1024,
      stop: ['</s>']
    },
    true
  );
}

runIfCalledAsScript(main, import.meta.url);
