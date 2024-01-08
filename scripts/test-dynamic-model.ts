import {
  awaitJob,
  completion,
  enqueueAddModel,
  runIfCalledAsScript
} from 'scripts/shared';

async function main() {
  const modelName = 'microsoft/Orca-2-13b';
  const containerType = 'VllmContainerA100_40G';

  console.log(`Test adding model ${modelName}`);
  const body = await enqueueAddModel(modelName);
  console.log('Successfully queued model to add', body);

  // fetch job success or failure for up to the timeout of 1 hour
  const timeoutMs = 60 * 60 * 1000;
  await awaitJob(body.job_id, timeoutMs);

  console.log('Model added successfully');

  const prompt = `What was Project A119 and what were its objectives?`;

  console.log(`Testing prompt: ${prompt}`);
  await completion(prompt, {
    model: modelName,
    max_tokens: 1024,
    stop: ['</s>'],
    stream: false,
    container: containerType
  });
}

runIfCalledAsScript(main, import.meta.url);
