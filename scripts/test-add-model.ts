import assert from 'assert';
import { getApiUrl, getAuthHeaders, runIfCalledAsScript } from 'scripts/shared';

async function main(model?: string) {
  const modelName = model || process.env.MODEL!;
  console.log(`Test adding model ${modelName}`);
  const payload = {
    name: modelName
  };

  const response = await fetch(getApiUrl('/models'), {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(payload)
  });

  assert(response.ok, 'Failed to add model: ' + response.status);
  const body = await response.json();
  console.log('Successfully queued model to add', body);

  // fetch until 200 or failure for up to the timeout of 1 hour
  const timeout = 60 * 60 * 1000;
  const start = Date.now();
  const end = start + timeout;
  while (Date.now() < end) {
    const statusResponse = await fetch(getApiUrl(`/jobs/${body.job_id}`), {
      headers: getAuthHeaders()
    });
    if (statusResponse.status === 200) {
      console.log('Model added successfully');
      break;
    }
    if (statusResponse.status != 202) {
      throw new Error('Failed to add model: ' + statusResponse.status);
    }

    console.log('Model download still in progress...');

    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  console.log('Unauthorized request test');
  const unauthedResp = await fetch(getApiUrl('/models'), {
    method: 'POST',
    headers: getAuthHeaders('BADKEY'),
    body: JSON.stringify(payload)
  });

  if (unauthedResp.ok || unauthedResp.status !== 401) {
    throw new Error('Unauthorized request returned unexpected response');
  }
}

runIfCalledAsScript(main, import.meta.url);
