import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main(model?: string) {
  const prompt = `USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:`;

  console.log('Happy-path tests');
  await completion(prompt, {
    model,
    max_tokens: 1024,
    stop: ['</s>'],
    stream: true
  });

  await completion(prompt, {
    model,
    max_tokens: 1024,
    stop: ['</s>']
  });

  console.log('Unauthorized request test');
  const unauthedResp = await completion(prompt, {
    model,
    apiKey: 'BADKEY'
  });

  if (unauthedResp.ok || unauthedResp.status !== 401) {
    throw new Error('Unauthorized request returned unexpected response');
  }
}

runIfCalledAsScript(main, import.meta.url);
