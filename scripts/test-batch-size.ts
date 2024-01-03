import { getWordsFromFile } from 'scripts/get-words';
import { completion, runIfCalledAsScript } from 'scripts/shared';

const batchSize = 4;
const context = 16_000;
const inputSize = 12_000;
const wordsCount = Math.ceil((inputSize * 2) / 4);
const maxTokens = Math.floor(((context - inputSize) * 95) / 100);

async function main() {
  const inputs = await Promise.all(
    [...Array(batchSize)].map(() =>
      getWordsFromFile({
        wordsCount,
        startLine: Math.floor(Math.random() * 500)
      })
    )
  );

  await Promise.all(
    inputs.map((prompt) =>
      completion(prompt, {
        max_tokens: maxTokens
      })
    )
  );
}

runIfCalledAsScript(main, import.meta.url);
