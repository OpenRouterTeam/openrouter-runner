import { runIfCalledAsScript } from 'scripts/shared';

const listUrl = process.env.LIST_LORA_API_URL;
const createUrl = process.env.CREATE_LORA_API_URL;
const key = process.env.API_KEY;

async function listLora() {
  if (!listUrl || !key) {
    throw new Error('listLora: Missing url or key');
  }

  const listResponse = await fetch(listUrl, {
    headers: {
      Authorization: `Bearer ${key}`
    }
  });

  if (!listResponse.ok) {
    throw new Error(await listResponse.text());
  }

  const list = await listResponse.json();
  console.log(list);
}

async function createLora() {
  if (!createUrl || !key) {
    throw new Error('listLora: Missing url or key');
  }
  const createResponse = await fetch(createUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${key}`
    },
    body: JSON.stringify({})
  });

  if (createResponse.ok && createResponse.body) {
    const stream = createResponse.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { value, done } = await stream.read();
      if (done) {
        break;
      }
      console.log(decoder.decode(value));
    }
  } else {
    const output = await createResponse.text();
    console.error(output);
  }
}

async function main() {
  await listLora();
  await createLora();
  await listLora();
}

runIfCalledAsScript(main, import.meta.url);
