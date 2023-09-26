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
  } as Record<string, unknown>
) {
  if (!url || !key) {
    throw new Error('Missing url or key');
  }

  const bodyPayload: Record<string, unknown> = {
    id: Math.random().toString(36).substring(7),
    prompt,
    params
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

  if (p.ok) {
    const output = await p.json();
    console.log(output);
  } else {
    const output = await p.text();
    console.error(output);
  }
}
