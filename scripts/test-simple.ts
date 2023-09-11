import { completion } from 'scripts/shared';

async function main() {
  const prompt = 'What was Project A119 and what were its objectives?';

  await completion(prompt);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
