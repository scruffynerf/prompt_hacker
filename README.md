# prompt_hacker
Automatic 1111 script to mess with your prompt in a variety of ways

Inspired by Tokenizer and Test My Prompt [https://github.com/Extraltodeus/test_my_prompt]
As if they had a baby, who grew up to run wild and graffiti all over your precious prompts

Still under heavy construction - v0.8

Current features:
* Remove - remove the token
* Randomize - change the token to something random
* Shuffle - Shuffle the token (must 'take' at least 2 tokens, if only 1, does 2)
* Strengthen/Weaken - adds a parenthesis and colon and a strength of 0.1 to 2.0
* MORE?  Ideas welcomed.

* Can work on Prompt or Negative Prompt - dropdown
* Now support TI embeds, LORAs, weighted prompts, and even lets you set custom words - textfield
* Option to avoid split words into multiple tokens and treat it as one entity
* Can skip X tokens at start - slider
* Take more than 1 token at a time - slider
* Options to ignore punctuation marks or common words - checkboxes
* Power of Strength/Weakness - slider + negative value checkbox
* Grid options - radio buttons
