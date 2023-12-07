import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main(model?: string) {
  const prompt = `USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:`;

  await completion(prompt, {
    model,
    max_tokens: 1024,
    stop: ['</s>'],
    stream: true
  });
}

runIfCalledAsScript(main, import.meta.url);
