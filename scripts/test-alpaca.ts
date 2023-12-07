import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main() {
  const prompt = `Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
What was Project A119 and what were its objectives?

### Response:
`;

  await completion(prompt, {
    max_tokens: 1024,
    stop: ['###'],
    stream: true
  });
}

runIfCalledAsScript(main, import.meta.url);
