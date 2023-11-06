import { getWordsFromFile } from 'scripts/get-words';
import { completion, runIfCalledAsScript } from 'scripts/shared';

async function main() {
  const prompt = await getWordsFromFile({
    wordsCount: 2500
  });

  await completion(prompt, {
    max_tokens: 42
  });
}

runIfCalledAsScript(main, import.meta.url);
