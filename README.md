# prompt_hacker
Automatic 1111 script to mess with your prompt in a variety of ways

Inspired by Tokenizer and Test My Prompt [https://github.com/Extraltodeus/test_my_prompt]

Still under construction.

Current features:
* Remove - remove the token
* Randomize - change the token to something random
* Shuffle - Shuffle the token (must 'take' at least 2 tokens, if only 1, does 2)
* Strengthen/Weaken - adds a parenthesis and colon and a strength of 0.1 to 2.0

* Can work on Prompt or Negative Prompt - dropdown
* Support LORAs, either in front or end of prompt, or remove them - dropdown
* Can skip X tokens at start - slider
* Take more than 1 token at a time - slider
* Options to ignore punctuation marks, numbers, and common words - checkboxes
* Power of Strength/Weakness - slider + negative value checkbox
* Grid options - radio buttons
