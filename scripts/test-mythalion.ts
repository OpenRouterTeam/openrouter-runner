import { config } from 'dotenv';

config({ path: '.env.local' });

async function main() {
  const prompt = 'What was Project A119 and what were its objectives?';

  const p = await fetch(`${process.env.MYTHALION_URL}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${process.env.MYTHALION_API_KEY}`
    },
    body: JSON.stringify({
      id: Math.random().toString(36).substring(7),
      prompt,
      params: {}
    })
  });

  const output = await p.text();

  console.log(output);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
