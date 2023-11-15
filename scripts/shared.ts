import { realpathSync } from 'fs';
import { pathToFileURL } from 'url';
import { config } from 'dotenv';

const envFile = `.env.${process.argv[2] ?? 'dev'}`;

config({ path: envFile });

const url = process.env.API_URL;
const key = process.env.API_KEY;
const model = process.env.MODEL;

export async function completion(
  prompt: string,
  params = {
    max_tokens: 16
  } as Record<string, unknown>,
  stream = false
) {
  if (!url || !key) {
    throw new Error('Missing url or key');
  }

  const bodyPayload: Record<string, unknown> = {
    id: Math.random().toString(36).substring(7),
    prompt,
    params,
    stream
  };

  if (model) {
    bodyPayload.model = model;
  }

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
    console.error(output);
  }
}

export function isEntryFile(url: string) {
  const realPath = realpathSync(process.argv[1]!);
  const realPathURL = pathToFileURL(realPath);

  return url === realPathURL.href;
}

// Passs down import.meta.url from the caller
export function runIfCalledAsScript(fn: () => Promise<void>, url: string) {
  if (isEntryFile(url)) {
    fn().catch((error) => {
      console.error(error);
      process.exit(1);
    });
  }
}
