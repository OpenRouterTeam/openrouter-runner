import { once } from 'events';
import fs from 'fs/promises';
import { createInterface } from 'readline';

export async function getWords(n: number, fileNo = 0): Promise<string> {
  const filePath = new URL(`./${fileNo}.txt`, import.meta.url);
  const fileStream = await fs.open(filePath, 'r');
  const input = fileStream.createReadStream();

  const rl = createInterface({
    input,
    crlfDelay: Infinity
  });

  const words: string[] = [];

  const lineEvent = new Promise<void>((resolve) => {
    rl.on('line', (line) => {
      const lineWords = line.split(/\s+/).filter(Boolean);
      for (const word of lineWords) {
        if (words.length < n) {
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
