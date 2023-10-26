import { once } from 'events';
import fs from 'fs/promises';
import { createInterface } from 'readline';

export async function getWordsFromFile({
  wordsCount = Infinity,
  fileName = '0',
  startLine = 0
} = {}): Promise<string> {
  const filePath = new URL(`./${fileName}.txt`, import.meta.url);
  const fileStream = await fs.open(filePath, 'r');
  const input = fileStream.createReadStream();

  const rl = createInterface({
    input,
    crlfDelay: Infinity
  });

  const words: string[] = [];

  const lineEvent = new Promise<void>((resolve) => {
    let currentLine = 0;
    rl.on('line', (line) => {
      currentLine++;
      if (currentLine < startLine) {
        return;
      }
      const lineWords = line.split(/\s+/).filter(Boolean);
      for (const word of lineWords) {
        if (words.length < wordsCount) {
          words.push(word);
        } else {
          rl.removeAllListeners('line');
          resolve();
          break;
        }
      }
    });
  });

  const closeEvent = once(rl, 'close');

  await Promise.race([lineEvent, closeEvent]);

  await fileStream.close();

  return words.join(' ');
}
