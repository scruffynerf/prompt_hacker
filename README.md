# prompt_hacker
Automatic 1111 script to mess with your prompt in a variety of ways

Inspired by Tokenizer and Test My Prompt [https://github.com/Extraltodeus/test_my_prompt]
As if they had a baby, who grew up to run wild and graffiti all over your precious prompts

Still under heavy construction - v0.9 - Major rewrite/refactor...

Current features:
* Remove - remove one or more tokens, front or back of a group, or entire group
* Randomize - Similar to remove, but change the token to something random
* Custom Word - Similar to remove, but change the token to your choice
* Shuffle - Shuffle a group of tokens (must 'take' at least 2 tokens at a time, if only 1 picked, still does 2)
* Strengthen/Weaken - adds a parenthesis and colon and a strength of 0.1 to 2.0 (or negative)
* Before and After - adds [Before:After:When] (When is 0-1)
* MORE?  Ideas welcomed.

* Can work on Prompt or Negative Prompt - dropdown
* Now support TI embeds, LORAs, weighted prompts, and even lets you set "protected" words to not split - textfield
* Option to avoid split words into multiple tokens and treat it as one entity
* Choice of many different options of operation, you can pick more than one at a time...
* Custom Word - text field
* Can select where in the prompt to start and end by percentage - 2 sliders
* Process/group 1+ token(s) at a time - slider
* Options to ignore punctuation marks or common words - checkboxes
* Power of Strength/Weakness (also Before/After duration)) - slider + negative value checkbox
* Grid options - radio buttons
