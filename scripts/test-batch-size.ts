import { getWordsFromFile } from 'scripts/get-words';
import { completion, runIfCalledAsScript } from 'scripts/shared';

const batchSize = 8;
const context = 5_000;
const inputSize = (context * 3) / 4;
const maxTokens = context - inputSize;

async function main() {
  const inputs = await Promise.all(
    [...Array(batchSize)].map(() =>
      getWordsFromFile({
        wordsCount: inputSize,
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
