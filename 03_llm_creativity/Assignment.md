Skip to the content.
Social Algorithms
Yale University, S&DS 3350/5350, Spring 2026
Overview Detailed Syllabus Assignments
Assignment 3: AI and creativity
Due Date: Monday 3/2 11:59p

Overview
In this assignment, you’ll build a small empirical replication of key ideas in Competition and Diversity in Generative AI (Raghavan, 2024), using LLMs to play Scattergories. The goal is to build an understanding of the role of randomness in diversifying LLM behavior, and the consequences in competitive environments.

You will study creativity and diversity in two ways:

Within a model: how sampling parameters (especially temperature, also top-k) change output distributions
In competition: how competition between different models unfolds when creativity is on the line
The assignment builds in stages from a single prompt and one model, to self-play, to cross-model competition.

You should read the introduction of the paper before you begin. You may also be interested in Raghavan’s other papers on ‘algorithmic monoculture’.

Required Setup
You will need to use:

Ollama, a tool for running local LLM models, with at least three different local player models
Calls to an OpenAI model for the separate judge script (only)
You are provided a Scattergories question bank:

assets/assignment3/scattergories_questions.csv
Two-phase workflow
Your implementation should follow this structure:

Player phase (local): generate answers to questions and write them to CSV files.
Judge phase (OpenAI API): run a separate judge script over one or more answer files to validate answers and compute points. Feel free to use this script during any phase of the assignment.
You do not need to interleave player generation and judging in one gameplay loop. Just generate the outputs from the different models separately (into separte CSVs), and have the answers compete against each other using the judge. The goal is to keep focus on experimenting with how LLM players behave.

Answer file format
Your player outputs should be CSV files with at least these columns:

question_id
letter
category
round_idx
answer
Each row should represent one player answer for one (question_id, round_idx) pair.

You may include extra metadata columns (for example: model, temperature, top_k, prompt_id) if helpful for your later analysis.

Judge script interface
Use the standalone judge script, though note it needs your OPENAI_API_KEY. For example:

python3 judge.py answers_modelA.csv
python3 judge.py answers_modelA.csv answers_modelB.csv answers_modelC.csv --out scores.csv --details judged_rows.csv
Behavior of the judge script:

Accepts 1 or more input answer files.
Judges each answer for validity with calls to gpt-5-mini with temperature=0.
Award points by uniqueness among submitted files for each (question_id, round_idx):
score = 1 if answer is valid and not duplicated by another submitted player on that round
else score = 0
Write score outputs to CSV (at least one summary CSV).
It caches judge calls by normalized key (letter, category, answer_normalized) to reduce repeated cost.

The judge code normalizes answers before judging. It should:
convert to lowercase
collapse whitespace
strip edge punctuation
Install Ollama
Install Ollama. If you need help, LLMs are very good at debugging and explaining how to set up Ollama.

Mac:

Download installer: https://ollama.com/download/mac
Install and open the Ollama app once.
In Terminal, verify with ollama --version.
Windows:

Download installer: https://ollama.com/download/windows
Install and open Ollama once.
In PowerShell, verify with ollama --version.
Local player models
Minimum requirement:

Choose at least 3 different local models.
If your hardware supports it, include at least 1 model in the 7B-8B range.
If it does not, use the fallback list below and document your constraints in your report.
Standard (most recent laptops with enough RAM):

ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.2:3b
ollama pull gemma2:2b
Fallback (resource-constrained machines):

ollama pull llama3.2:1b
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:3b
ollama pull gemma2:2b
If your machine is stronger, optional larger models include:

qwen2.5:7b
llama3.1:8b
mistral:7b
gemma2:9b
Cost control
When working with the judge, set hard spend limits on your OpenAI API account before running large sweeps.
Run a small pilot first (for example 5 questions, 10 rounds) before full judging.
I (think) I’ve successfully written the judge to cache outputs, which also should make it faster.
Report bugs please.
Starter code
Starter code is provided on github:

assets/assignment3/assignment3_starter.py
assets/assignment3/judge.py
assets/assignment3/scattergories_questions.csv
Quick start:

python3 -m venv .venv && source .venv/bin/activate
Set API key in judge.py (or set it as part of your env, export OPENAI_API_KEY=...).
Generate one or more answer CSV files from your player experiments (for example with assignment3_starter.py generate-answers, or your own script/notebook).
Run judging/scoring as a separate pass:
python3 judge.py outputs/answers_modelA.csv outputs/answers_modelB.csv --out outputs/scores.csv --details outputs/judged_rows.csv
for single-file evaluation: python3 judge.py outputs/answers_modelA.csv --out outputs/scores_single.csv
Important:

The scaffold is intentionally incomplete.
You are expected to improve player prompts/policies and add analyses/plots.
Treat starter outputs as a baseline, not a final submission.
Part I: Single-Model Calibration
I.1 Warm-up: one simple Scattergories question
Use a single local Ollama player model first.

Task:

Use the question: “Name a day of the week.” Design a prompt to have the LLM produce (and only produce) the proposed answer. Your goal is to make the distribution of outcomes as close to uniform as possible over 7 choices, without producing invalid answers (answers that aren’t days of the week).
Sample 500 generations per temperature over a grid of temperatures. Your grid should include both low temperatures and very high temperatures.
Plot histograms of answer frequencies at each temperature.
Report some measure of variability (e.g., entropy) and describe the steps you took to produce something relatively close to uniform, and the challenges you faced. As a fair warning, it is quite hard to get the LLM to make it’s choices close to uniformly at random. Even getting positive probabilities on all days of the week counts as an acheivement.

I.2 Next-token probabilities and top-k
For this day-of-week setup:

For models where top-k is available, ensure k >= 7 , so all seven days can remain reachable.
Try to expose token probabilities/logprobs (typically via direct Ollama API calls), collect top next-token probabilities, and compare them to the histograms from above.
If your chosen model/tooling path does not expose token logprobs, document that clearly and proceed with histogram-only analysis.
I.3 Harder prompt
Now repeat the histogram exercise with a harder Scattergories question:

“Fruits that start with b”
Task:

Keep the same overall experimental pipeline, but now explore prompting instructions for this harder game category. Experiment with different prompts and temperatures. Does it help or hurt to tell the LLM that you are asking it to play Scattergories?
Sample 500 generations per temperature. Include the same very-high-temperature range here, as well as low temperature.
Build histograms and compute diversity metrics.
Compare how challenges with answer validity and uniformity differ from the day-of-week question.
Part II: Self-Play with One Model
Now let one model play against itself in a 2-player Scattergories game with many questions.

Use the provided question bank:

assets/assignment3/scattergories_questions.csv
II.1 Game definition
Use a two-phase process.

Phase A: generate answers

For each row (letter, category) and each round index, have the player output one answer. Write a prompt that wraps around the category and letter from scattergories_questions.csv, and set the temperature informed by your experiences above. Run both instances of the model (player 1 and player 2) with the same prompt and same temperature (optional: explore varying temperature here as well).
Write the answers to a CSV file in the required format.
Keep this generation step independent from judging.
Phase B: judge and score

Run judge.py on one or more answer files. Reminder that will require your OpenAI API key.
The judge script will:
call a GPT judge for validity (yes/no)
normalize answers
compute points across submitted player files
output score CSV
Audit quality: randomly sample at least 50 judged examples and manually verify them; report estimated judge error rate.
II.2 Self-play experiments
Once you have generation and judging figured out, run repeated rounds for each question (enough rounds for stable estimates (of the expected score) and store generated answers.
Run judge.py on your generated files to compute game outcomes from self-play.
Measure per-question and overall outcomes:
Validity rate
Average score per player
Revisit prompt/temperature choices and see whether you can improve the self-play score.
Part III: Cross-Model Competition
Now evaluate the role of diversity across player models.

III.1 Local model set
Use at least 3 different local models. Run pairwise 2-player competitions for your chosen local models.

Generate one answer file per player/model. Use the same prompt and temperature settings across the models (optional: explore variations in prompt/temperature).
Judge each matchup by passing the relevant files to judge.py.
Use the returned CSV scores for analysis.
III.2 Analysis
Compare self-play for each local model and cross-play for each pair of local models.

Report at least:

Average score per player against each opponent
Validity rates for each model
Discuss what it means for models to do differently well against themselves than they do against other models, and the role of temperature in that behavior. Hint: what happens when a low temperature model plays against itself?

Deliverables
Submit the following:

Report (PDF)
Stepping through the requested output and responses in Parts I, II, and III above
Short responses to the reflection questions below
Code and output data
scripts/notebooks used to run all experiments.
CSV/JSON sufficient to reproduce key figures/tables, including
player-generated answer CSV files
judge output CSV files (scores and/or row-level judgments)
Reflection Questions
Include concise answers to these in your report:

For the day-of-week task, what prompt and sampling settings did you use, and how close did you get to uniform over valid answers? (Getting close to uniform was surprisingly hard, I found, so it’s OK if you can’t get close to uniform.)
Were you able to extract next-token logprobs? If yes, how closely did they match empirical frequencies? If no, what blocked it?
For “fruits that start with b,” which prompt variants did you test, and did explicitly framing it as Scattergories help or hurt?
After revising prompt/temperature, how much did self-play performance change, and what do you think caused the change?
How did self-play outcomes differ from cross-model outcomes, and what does that imply about diversity across models?
What role did temperature play in self-play vs cross-play performance?
Overall, how much of your performance gain came from better prompts vs better sampling settings?
What do these experiments suggest about AI systems interacting with other AI systems in the wild?
What follow-up experiment would you run next, and why?
Grading Rubric
Component	Weight	Criteria
Part I: Calibration + distributions	25%	Sound experiment design, clear histograms, correct metrics, and explicit high-temperature evaluation
Part I: Probability comparison + top-k discussion	15%	Correct top-k analysis, correct interpretation of API/tooling capabilities, and careful probability/frequency comparison
Part II: Self-play generation pipeline + analysis	25%	Correct answer-file generation, correct use of judge outputs, stable estimates, prompt/temperature retuning, and credible judge-audit analysis
Part III: Cross-model competition analysis	25%	Clear cross-model comparison using at least 3 local models, with evidence-based conclusions about diversity
Workflow and compliance requirements	5%	Correct two-phase workflow usage, required deliverables/files (including prompts), cost-control/accounting items, complete reflection responses, and academic-integrity disclosures
Code and report quality	5%	Reproducible, readable code; clear writing; and clear connection to the Raghavan paper
Technical references
OpenAI Chat Completions API docs: https://platform.openai.com/docs/api-reference/chat/create
OpenAI logprobs notebook: https://cookbook.openai.com/examples/using_logprobs
Ollama docs: https://github.com/ollama/ollama
Ollama model library: https://ollama.com/library
Recommended model pages:
https://ollama.com/library/qwen2.5
https://ollama.com/library/llama3.1
https://ollama.com/library/mistral
https://ollama.com/library/gemma2
Academic Integrity
You may work in groups of up to 3 students. List all group members in your report and code. You may use AI tools, but you must briefly describe what you used them for.

Good luck!

This page was generated by GitHub Pages.