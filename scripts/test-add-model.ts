import {
  defaultModel,
  enqueueAddModel,
  getApiUrl,
  getAuthHeaders,
  pollForJobCompletion,
  runIfCalledAsScript
} from 'scripts/shared';

async function main(modelName = defaultModel) {
  console.log(`Test adding model ${modelName}`);
  const body = await enqueueAddModel(modelName);
  console.log('Successfully queued model to add', body);

  // fetch job success or failure for up to the timeout of 1 hour
  const timeoutMs = 60 * 60 * 1000;
  await pollForJobCompletion(body.job_id, timeoutMs);

  console.log('Model added successfully');

  console.log('Unauthorized request test');
  const unauthedResp = await fetch(getApiUrl('/models'), {
    method: 'POST',
    headers: getAuthHeaders('BADKEY'),
    body: JSON.stringify({
      name: modelName
    })
  });

  if (unauthedResp.ok || unauthedResp.status !== 401) {
    throw new Error('Unauthorized request returned unexpected response');
  }
}

runIfCalledAsScript(main, import.meta.url);
