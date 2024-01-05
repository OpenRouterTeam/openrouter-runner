import { getWordsFromFile } from 'scripts/get-words';
import { completion, runIfCalledAsScript } from 'scripts/shared';

const batchSize = 8;
// const context = 8_000;
const wordsCount = 4_000;
const maxTokens = 2_000;

async function main() {
  const inputs = await Promise.all(
    [...Array(batchSize)].map(() =>
      getWordsFromFile({
        wordsCount,
        startLine: Math.floor(Math.random() * 500)
      })
    )
  );

  await Promise.allSettled(
    inputs.map((prompt) =>
      completion(prompt, {
        stream: true,
        max_tokens: maxTokens
      })
    )
  );
}

runIfCalledAsScript(main, import.meta.url);
