import {
  completion,
  defaultModel,
  enqueueAddModel,
  pollForJobCompletion,
  runIfCalledAsScript
} from 'scripts/shared';

async function main(
  modelName = defaultModel
) {
  console.log(`Test adding model ${modelName}`);
  const body = await enqueueAddModel(modelName);
  console.log('Successfully queued model to add', body);

  // fetch job success or failure for up to the timeout of 1 hour
  const timeoutMs = 60 * 60 * 1000;
  await pollForJobCompletion(body.job_id, timeoutMs);

  console.log('Model added successfully');

  const prompt = `What was Project A119 and what were its objectives?`;

  console.log(`Testing prompt: ${prompt}`);
  await completion(prompt, {
    model: modelName,
    max_tokens: 1024,
    stop: ['</s>'],
    stream: false
  });
}

runIfCalledAsScript(main, import.meta.url);
