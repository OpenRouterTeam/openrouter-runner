import { config } from 'dotenv';

const envFile = `.env.${process.argv[2] ?? 'dev'}`;

config({ path: envFile });

const url = process.env.API_URL;
const key = process.env.API_KEY;

export async function completion(
  prompt: string,
  params = {
    max_tokens: 16
  }
) {
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

  if (p.ok) {
    const output = await p.json();
    console.log(output);
  } else {
    const output = await p.text();
    console.error(output);
  }
}
