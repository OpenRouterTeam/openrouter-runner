import { getWords } from 'scripts/get-words';
import { completion } from 'scripts/shared';

async function main() {
  const prompt = await getWords(3900);

  await completion(prompt);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
