import { realpathSync } from 'fs';
import { pathToFileURL } from 'url';
import { config } from 'dotenv';

const envFile = `.env.dev`;

config({ path: envFile });

const url = process.env.API_URL!;
const key = process.env.RUNNER_API_KEY!;
const defaultModel = process.env.MODEL;
const defaultContainer = process.env.CONTAINER_TYPE;

export function getApiUrl(path: string) {
  return `${url}${path}`;
}

export function getAuthHeaders(apiKey = key) {
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
    apiKey = key,
    quiet = false,
    container = defaultContainer
  } = {}
) {
  if (!quiet) {
    console.info(`Calling ${url} with model ${model}, stream: ${stream}`);
  }

  let bodyPayload: Record<string, unknown> = {
    id: Math.random().toString(36).substring(7),
    prompt,
    model,
    params: { max_tokens, stop },
    stream
  };

  if (container) {
    bodyPayload['runner'] = { container };
  }

  const p = await fetch(getApiUrl(''), {
    method: 'POST',
    headers: getAuthHeaders(apiKey),
    body: JSON.stringify(bodyPayload)
  });

  const output = p.ok && !stream ? await p.json() : await p.text();
  if (!quiet) {
    console.log(output);
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

export async function awaitJob(jobId: string, timeoutMs: number) {
  const start = Date.now();
  const end = start + timeoutMs;
  while (Date.now() < end) {
    const statusResponse = await fetch(getApiUrl(`/jobs/${jobId}`), {
      headers: getAuthHeaders()
    });
    if (statusResponse.status === 200) {
      console.log('Job completed successfully');
      break;
    }
    if (statusResponse.status != 202) {
      throw new Error('Failed to process job: ' + statusResponse.status);
    }

    console.log('Job still in progress...');

    await new Promise((resolve) => setTimeout(resolve, 5000));
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
