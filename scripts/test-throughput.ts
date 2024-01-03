import { getWordsFromFile } from 'scripts/get-words';
import { completion, runIfCalledAsScript } from 'scripts/shared';

const prompts = [
  `Ullamco mollit deserunt mollit ea est cillum nulla et est. Magna incididunt laborum amet consequat sint amet. Cillum aliqua id nulla nisi in. Sunt commodo aliqua quis deserunt. Adipisicing eu officia laboris commodo. Ullamco sunt irure eu aute laborum ut commodo ullamco incididunt duis officia cupidatat et voluptate.

  Deserunt Lorem minim ex voluptate amet et qui duis irure. Mollit occaecat est duis sunt minim do non cupidatat labore tempor consectetur cupidatat occaecat et. Nulla irure labore proident aliqua dolore cupidatat officia ut anim laborum fugiat culpa est eu.
  
  Ex irure aute minim proident aute amet deserunt minim sint dolore. Fugiat occaecat esse incididunt aliqua. Anim incididunt nulla mollit consequat anim laborum amet. Non aliquip ex sint eu dolore non ea magna nostrud ea et.
  
  Culpa laborum ipsum enim ex nisi officia ullamco elit consequat ad. Nulla dolore nisi mollit do quis dolor. Lorem culpa irure magna esse officia esse ex amet ea tempor proident eu do consectetur. Aute nulla proident veniam ea. Deserunt minim proident laboris ipsum anim elit cupidatat sit adipisicing aliqua cillum aute magna qui. Dolore voluptate pariatur dolore quis Lorem quis Lorem id aliquip.
  
  Laborum nisi pariatur fugiat est aliqua aliquip ad minim duis et. Elit et sint officia nisi. Qui amet in ipsum ipsum adipisicing sint aliquip id cillum commodo consectetur deserunt nisi. Sunt mollit occaecat exercitation deserunt. Ea duis commodo do mollit nostrud cupidatat.
  
  Sunt labore non deserunt laboris eiusmod laboris amet laborum sit sunt. Quis eiusmod fugiat do ipsum anim sit mollit non occaecat ullamco excepteur ea reprehenderit. Pariatur reprehenderit est eu do deserunt aliquip cillum consequat id laborum.
  
  Quis quis deserunt in officia id non irure anim voluptate. Ad elit sint ex aliqua in in commodo commodo consectetur ipsum incididunt deserunt enim. Officia ut cillum officia elit anim aute eu. Lorem sit ut incididunt irure do sint non eu consequat commodo aute esse. Laboris Lorem nisi proident sunt amet consequat magna amet ex ipsum. Tempor et consectetur exercitation id sint minim culpa deserunt proident tempor duis est cupidatat. Quis ea veniam pariatur proident.`,
  'Implement a Python function to compute the Fibonacci numbers.',
  'Write a Rust function that performs binary exponentiation.',
  'How do I allocate memory in C?',
  'What are the differences between Javascript and Python?',
  'How do I find invalid indices in Postgres?',
  'How can you implement a LRU (Least Recently Used) cache in Python?',
  'What approach would you use to detect and prevent race conditions in a multithreaded application?',
  'Can you explain how a decision tree algorithm works in machine learning?',
  'How would you design a simple key-value store database from scratch?',
  'How do you handle deadlock situations in concurrent programming?',
  'What is the logic behind the A* search algorithm, and where is it used?',
  'How can you design an efficient autocomplete system?',
  'What approach would you take to design a secure session management system in a web application?',
  'How would you handle collision in a hash table?',
  'How can you implement a load balancer for a distributed system?',
  'What is the fable involving a fox and grapes?',
  'Write a story in the style of James Joyce about a trip to the Australian outback in 2083, to see robots in the beautiful desert.',
  'Who does Harry turn into a balloon?',
  "Write a tale about a time-traveling historian who's determined to witness the most significant events in human history.",
  "Describe a day in the life of a secret agent who's also a full-time parent.",
  'Create a story about a detective who can communicate with animals.',
  'What is the most unusual thing about living in a city floating in the clouds?',
  'In a world where dreams are shared, what happens when a nightmare invades a peaceful dream?',
  'Describe the adventure of a lifetime for a group of friends who found a map leading to a parallel universe.',
  'Tell a story about a musician who discovers that their music has magical powers.',
  'In a world where people age backwards, describe the life of a 5-year-old man.',
  'Create a tale about a painter whose artwork comes to life every night.',
  "What happens when a poet's verses start to predict future events?",
  'Imagine a world where books can talk. How does a librarian handle them?',
  'Tell a story about an astronaut who discovered a planet populated by plants.',
  'Describe the journey of a letter traveling through the most sophisticated postal service ever.',
  "Write a tale about a chef whose food can evoke memories from the eater's past.",
  'What were the major contributing factors to the fall of the Roman Empire?',
  'How did the invention of the printing press revolutionize European society?',
  'What are the effects of quantitative easing?',
  'How did the Greek philosophers influence economic thought in the ancient world?',
  'What were the economic and philosophical factors that led to the fall of the Soviet Union?',
  'How did decolonization in the 20th century change the geopolitical map?',
  "What was the influence of the Khmer Empire on Southeast Asia's history and culture?",
  'Describe the city of the future, considering advances in technology, environmental changes, and societal shifts.',
  'In a dystopian future where water is the most valuable commodity, how would society function?',
  'If a scientist discovers immortality, how could this impact society, economy, and the environment?',
  'What could be the potential implications of contact with an advanced alien civilization?',
  'What is the product of 9 and 8?',
  'If a train travels 120 kilometers in 2 hours, what is its average speed?',
  'Think through this step by step. If the sequence a_n is defined by a_1 = 3, a_2 = 5, and a_n = a_(n-1) + a_(n-2) for n > 2, find a_6.',
  'Think through this step by step. Calculate the sum of an arithmetic series with first term 3, last term 35, and total terms 11.',
  'Think through this step by step. What is the area of a triangle with vertices at the points (1,2), (3,-4), and (-2,5)?',
  'Think through this step by step. Solve the following system of linear equations: 3x + 2y = 14, 5x - y = 15.',
  "Who was Emperor Norton I, and what was his significance in San Francisco's history?",
  'What is the Voynich manuscript, and why has it perplexed scholars for centuries?',
  'What was Project A119 and what were its objectives?',
  "What is the 'Dyatlov Pass incident' and why does it remain a mystery?",
  "What is the 'Emu War' that took place in Australia in the 1930s?",
  "What is the 'Phantom Time Hypothesis' proposed by Heribert Illig?",
  "Who was the 'Green Children of Woolpit' as per 12th-century English legend?",
  "What are 'zombie stars' in the context of astronomy?",
  "Who were the 'Dog-Headed Saint' and the 'Lion-Faced Saint' in medieval Christian traditions?",
  "What is the story of the 'Globsters', unidentified organic masses washed up on the shores?"
];

async function main() {
  const veryRandomPrompt = await getWordsFromFile({
    wordsCount: 2000,
    startLine: Math.floor(Math.random() * 500)
  });

  await Promise.all(
    [...prompts, veryRandomPrompt].map((p) =>
      completion(p, {
        max_tokens: 1000,
        quiet: true
      })
    )
  );
}

runIfCalledAsScript(main, import.meta.url);
