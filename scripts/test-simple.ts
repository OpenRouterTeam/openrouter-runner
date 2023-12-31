import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main(model?: string) {
  const prompt = `USER: What was Project A119 and what were its objectives?\n\n ASSISTANT:`;

  await completion(prompt, {
    model,
    max_tokens: 1024,
    stop: ['</s>'],
    stream: true
  });

  await completion(prompt, {
    model,
    max_tokens: 1024,
    stop: ['</s>'],
  });

  // Unauthorized requests should fail with a 401
  let gotExpectedError = false;
  try {
      await completion(prompt, {model, apiKey: "BADKEY"});
  } catch (e: any) {
    gotExpectedError = e.message == "Status: 401";
  }
  if (!gotExpectedError) {
    throw new Error("Unauthorized request returned unexpected response")
  }
}

runIfCalledAsScript(main, import.meta.url);
