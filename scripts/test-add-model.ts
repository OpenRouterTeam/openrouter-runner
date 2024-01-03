import assert from 'assert';
import { postToApi, runIfCalledAsScript } from 'scripts/shared';

async function main(model?: string) {
  const modelName = model || process.env.MODEL!;
  console.log(`Test adding model ${modelName}`);
  const payload = {
    name: modelName
  };

  const response = await postToApi('/models', payload);

  assert(response.ok, 'Failed to add model: ' + response.status);

  console.log('Unauthorized request test');
  const unauthedResp = await postToApi('/models', payload, {
    apiKey: 'BADKEY'
  });

  if (unauthedResp.ok || unauthedResp.status !== 401) {
    throw new Error('Unauthorized request returned unexpected response');
  }
}

runIfCalledAsScript(main, import.meta.url);
