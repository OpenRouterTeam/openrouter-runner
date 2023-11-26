import { getWordsFromFile } from 'scripts/get-words';
import { runIfCalledAsScript } from 'scripts/shared';

const url = process.env.PUNC_API_URL;
const key = process.env.RUNNER_API_KEY;

async function main() {
  if (!url || !key) {
    throw new Error('punctuator: Missing url or key');
  }

  const input = await getWordsFromFile({
    fileName: 'steve-job-speech'
  });

  console.log('Input size: ', input.length);

  const t0 = Date.now();
  const createResponse = await fetch(url, {
    method: 'POST',
    keepalive: true,
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`
    },
    body: JSON.stringify({
      input
    })
  });

  if (createResponse.ok && createResponse.body) {
    const json = await createResponse.json();
    const t1 = Date.now();

    const duration = t1 - t0;
    console.log(json);
    console.log('Output size: ', json.text.length);

    console.log(`Duration: ${duration}ms`);
  } else {
    const output = await createResponse.text();
    console.error(output);
  }
}

runIfCalledAsScript(main, import.meta.url);
