import { config } from 'dotenv';

const envFile = `.env.${process.argv[2] ?? 'dev'}`;

config({ path: envFile });

const url = process.env.MYTHALION_URL;
const key = process.env.MYTHALION_API_KEY;

export async function completion(prompt: string, params = {}) {
  if (!url || !key) {
    throw new Error('Missing url or key');
  }

  const p = await fetch(url, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`
    },
    body: JSON.stringify({
      id: Math.random().toString(36).substring(7),
      prompt,
      params
    })
  });

  const output = await p.text();

  console.log(output);
}
