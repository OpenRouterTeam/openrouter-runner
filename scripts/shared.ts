import { realpathSync } from 'fs';
import { pathToFileURL } from 'url';
import { config } from 'dotenv';

const envFile = `.env.dev`;

config({ path: envFile });

const url = process.env.API_URL;
const key = process.env.RUNNER_API_KEY;
const defaultModel = process.env.MODEL;

export async function completion(
  prompt: string,
  {
    model = defaultModel,
    max_tokens = 16,
    stream = false,
    stop = ['</s>']
  } = {}
) {
  if (!url || !key) {
    throw new Error('Missing url or key');
  }

  const bodyPayload: Record<string, unknown> = {
    id: Math.random().toString(36).substring(7),
    prompt,
    model,
    params: { max_tokens, stop },
    stream
  };

  const p = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`
    },
    body: JSON.stringify(bodyPayload)
  });

  if (p.ok && !stream) {
    const output = await p.json();
    console.log(output);
  } else {
    const output = await p.text();
    console.log(output);
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
