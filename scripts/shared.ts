import { realpathSync } from 'fs';
import { pathToFileURL } from 'url';
import { config } from 'dotenv';

const envFile = `.env.dev`;

config({ path: envFile });

export const defaultModel = process.env.MODEL || 'microsoft/phi-2';

export function getApiUrl(path: string) {
  const url = process.env.API_URL;
  if (!url) {
    throw new Error('Missing API_URL');
  }

  return `${url}${path}`;
}

export function getAuthHeaders(apiKey = process.env.RUNNER_API_KEY) {
  if (!apiKey) {
    throw new Error('Missing RUNNER_API_KEY');
  }

  return {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`
  };
}

export async function completion(
  prompt: string,
  {
    model = defaultModel,
    max_tokens = 16,
    stream = false,
    stop = ['</s>'],
    apiKey = undefined as string | undefined,
    quiet = false
  } = {}
) {
  const apiUrl = getApiUrl('');
  if (!quiet) {
    console.info(`Calling ${apiUrl} with model ${model}, stream: ${stream}`);
  }

  const bodyPayload: Record<string, unknown> = {
    id: Math.random().toString(36).substring(7),
    prompt,
    model,
    params: { max_tokens, stop },
    stream
  };

  const p = await fetch(apiUrl, {
    method: 'POST',
    headers: getAuthHeaders(apiKey),
    body: JSON.stringify(bodyPayload)
  });

  const output = await p.text();
  if (!quiet) {
    console.log(`Response status: ${p.status}`);
    console.log('Output -------------------');
    console.log(output.trim());
  }

  return p;
}

export async function enqueueAddModel(modelName: string) {
  const payload = {
    name: modelName
  };

  const response = await fetch(getApiUrl('/models'), {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error('Failed to post model: ' + response.status);
  }

  return await response.json();
}

export async function pollForJobCompletion(
  jobId: string,
  timeoutMs: number,
  pollingIntervalMs = 5000
) {
  const start = Date.now();
  const end = start + timeoutMs;
  const url = getApiUrl(`/jobs/${jobId}`);
  const headers = getAuthHeaders();
  while (Date.now() < end) {
    const statusResponse = await fetch(url, {
      headers
    });
    if (statusResponse.status === 200) {
      console.log('Job completed successfully');
      break;
    }
    if (statusResponse.status !== 202) {
      throw new Error('Failed to process job: ' + statusResponse.status);
    }

    console.log('Job still in progress...');

    await new Promise((resolve) => setTimeout(resolve, pollingIntervalMs));
  }
}

export function isEntryFile(url: string) {
  const realPath = realpathSync(process.argv[1]!);
  const realPathURL = pathToFileURL(realPath);

  return url === realPathURL.href;
}

// Passs down import.meta.url from the caller
export function runIfCalledAsScript(
  fn: (...args: string[]) => Promise<void>,
  url: string
) {
  if (isEntryFile(url)) {
    // Call fn with the arguments passed in from the command line
    fn(...process.argv.slice(2)).catch((error) => {
      console.error(error);
      process.exit(1);
    });
  }
}
