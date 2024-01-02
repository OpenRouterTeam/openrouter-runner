import assert from 'assert';
import { postToApi, runIfCalledAsScript } from 'scripts/shared';

async function main(model?: string) {
  const modelName = model || process.env.MODEL!;
  console.log(`Test adding model ${modelName}`);

  const response = await postToApi('/models', {
    name: modelName
  });

  assert(response.ok, 'Failed to add model: ' + response.status);
}

runIfCalledAsScript(main, import.meta.url);
