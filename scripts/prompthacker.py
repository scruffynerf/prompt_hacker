# Inspired by (and started out using concepts/code from) 'Test My Prompt' and 'Tokenizer'
# Purpose: to rip your prompt into small bits and then staple/tape/pound it back to together in new and interesting ways...
#
# Authored by Scruffynerf  - msg me @Scruffy in the Civitai Discord

from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
import modules.scripts as scripts
from modules import script_callbacks, shared, sd_hijack
from PIL import Image, ImageFont, ImageDraw, ImageOps
from fonts.ttf import Roboto
import gradio as gr
from collections import namedtuple
from random import randint, sample
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
import re
import random

class VanillaClip:
    def __init__(self, clip):
        self.clip = clip

    def vocab(self):
        return self.clip.tokenizer.get_vocab()

    def byte_decoder(self):
        return self.clip.tokenizer.byte_decoder

class OpenClip:
    def __init__(self, clip):
        self.clip = clip
        self.tokenizer = open_clip.tokenizer._tokenizer

    def vocab(self):
        return self.tokenizer.encoder

    def byte_decoder(self):
        return self.tokenizer.byte_decoder

class Script(scripts.Script):
    GridSaveFlags = namedtuple('GridSaveFlags', ['never_grid', 'always_grid', 'always_save_grid'], defaults=(False, False, False))
    grid_options_mapping = {
        "Use user settings": GridSaveFlags(),
        "Don't generate": GridSaveFlags(never_grid=True),
        "Generate": GridSaveFlags(always_grid=True),
        "Generate and always save": GridSaveFlags(always_grid=True, always_save_grid=True),
        }
    default_grid_opt = list(grid_options_mapping.keys())[-1]

    def title(self):
        return "Prompt Hacker"

    def ui(self, is_img2img):
        with gr.Blocks():
            with gr.Box():
                neg_pos = gr.Dropdown(label="Hack apart the positive or negative prompt?", choices=["Positive","Negative"], value="Positive")
            with gr.Row():
                startat = gr.Slider(minimum=0, maximum=100, step=1, label='Start processing at % of prompt - 0%=start', value=0)
                endat = gr.Slider(minimum=0, maximum=100, step=1, label='Finish processing at % of prompt - 100%=end', value=100)
            with gr.Box():
                take_x_atonce = gr.Slider(minimum=1, maximum=100, step=1, label='Process X words/tokens at a time, as a group', value=1)
                nosplitwords = gr.Checkbox(label='Keep complex words whole (versus splitting into 2+ tokens)', value=True)
                ignorepunct = gr.Checkbox(label='Ignore+Remove punctuation like commas or periods', value=True)
                ignorecommon = gr.Checkbox(label="Ignore+Remove common words like 'a' 'an' 'the' etc", value=False)
                magicwords = gr.Textbox(label="Special Word handling (if not autorecognized)",lines=1,value="yourcustomword,yourcustomword2")
                customword = gr.Textbox(label="Custom word - when you want to force a word in",lines=1,value="")
            # handle token removal functions, then token adding, then token replacement, then token reorganizing, then wrapping/weights
            with gr.Box():
                tokenremoval = gr.Dropdown(label="Word/Token removal",choices=["","Remove Word/Token/Group", "Remove First in Group", "Remove Last in Group"],value="")
                tokenadding = gr.Dropdown(label="Word adding", choices=["TBD"],value="TBD")
                tokenreplacement = gr.Dropdown(label="Word Replacement", choices=["","Change All to 1 Custom", "Change First to Custom", "Change Last to Custom", "Change All to 1 Random", "Change Each to Random", "Change First to Random", "Change Last to Random"], value="")
                tokenrearrange = gr.Dropdown(label="Rearranging Words", choices=["","Shuffle All"],value="")
            with gr.Box():
                tokenreweight = gr.Dropdown(label="Weights and Transitions", choices=["","Before and After", "Change Weight"], value="")
                with gr.Row():
                    power = gr.Slider(minimum=0, maximum=2, step=0.1, label='Stronger/Weaker weight (0-2) or Before/After (0-1) value', value=1)
                    powerneg = gr.Checkbox(label='make Negative value', value=False)
            with gr.Box():
                font_size = gr.Slider(minimum=12, maximum=64, step=1, label='Font size', value=32)
                grid_option = gr.Radio(choices=list(self.grid_options_mapping.keys()), label='Grid generation', value=self.default_grid_opt)
        return [neg_pos, startat, endat, take_x_atonce, nosplitwords, ignorepunct, ignorecommon, magicwords, customword, tokenremoval, tokenadding, tokenreplacement, tokenrearrange, tokenreweight, power, powerneg, font_size, grid_option]

    def run(self, p, neg_pos, startat, endat, take_x_atonce, nosplitwords, ignorepunct, ignorecommon, magicwords, customword, tokenremoval, tokenadding, tokenreplacement, tokenrearrange, tokenreweight, power, powerneg, font_size, grid_option):

        def write_on_image(img, msg):
            ix,iy = img.size
            draw = ImageDraw.Draw(img)
            margin=2
            fontsize=font_size
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(Roboto, fontsize)
            text_height=iy-60
            tx = draw.textbbox((0,0),msg,font)
            draw.text((int((ix-tx[2])/2),text_height+margin),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2),text_height-margin),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2+margin),text_height),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2-margin),text_height),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2),text_height), msg,(255,255,255),font=font)
            return img

        def vocabsize():
            clip = shared.sd_model.cond_stage_model.wrapped
            if isinstance(clip, FrozenCLIPEmbedder):
                clip = VanillaClip(shared.sd_model.cond_stage_model.wrapped)
            elif isinstance(clip, FrozenOpenCLIPEmbedder):
                clip = OpenClip(shared.sd_model.cond_stage_model.wrapped)
            else:
                raise RuntimeError(f'Unknown CLIP model: {type(clip).__name__}')
            vocab = {v: k for k, v in clip.vocab().items()}
            return len(vocab)

        def addnotreallytoken(word):
            notreallytokens.append(word)
            return (len(notreallytokens)-1)*-1 # returns the negative location of the word stored

        def tokenize(promptinput, input_is_ids=False):
            tokens = []

            if input_is_ids:
                # if we've already got token ids, use em
                tokens = promptinput
            else:
                # Just trusting the tokenizer code is no good, as it doesn't support embeddings, loras, parens/brackets/weights and some trigger words.
                splitintowords = promptinput.split(" ")
                for thisword in splitintowords:
                    commaonend = False
                    if thisword.rstrip(",") != thisword and thisword != ",":
                       # comma on the end
                       commaonend = True
                       thisword = thisword.rstrip(",")
                    if thisword in sd_hijack.model_hijack.embedding_db.word_embeddings.keys():
                        # this word is embedding
                        tokens.append(addnotreallytoken(thisword))
                    elif re.match("<.*>", thisword):
                        # this word is a lora
                        tokens.append(addnotreallytoken(thisword))
                    elif re.match("\(.*\)", thisword):
                        # this word is weighted
                        tokens.append(addnotreallytoken(thisword))
                    elif re.match("\[.*\]", thisword):
                        # this word is weighted
                        tokens.append(addnotreallytoken(thisword))
                    elif thisword in magicwords.split(","):
                        # this word is given to us to flag specifically
                        tokens.append(addnotreallytoken(thisword))
                    else:
                        # tokenize this
                        thistoken = shared.sd_model.cond_stage_model.tokenize([thisword])[0]
                        if nosplitwords and len(thistoken) > 1:
                           tokens.append(addnotreallytoken(thisword))
                        else:
                           tokens.extend(thistoken)
                    if commaonend:
                        tokens.append(267)

            #print(f"TOKENDEBUG {tokens}")

            clip = shared.sd_model.cond_stage_model.wrapped
            if isinstance(clip, FrozenCLIPEmbedder):
                clip = VanillaClip(shared.sd_model.cond_stage_model.wrapped)
            elif isinstance(clip, FrozenOpenCLIPEmbedder):
                clip = OpenClip(shared.sd_model.cond_stage_model.wrapped)
            else:
                raise RuntimeError(f'Unknown CLIP model: {type(clip).__name__}')
            vocab = {v: k for k, v in clip.vocab().items()}
            byte_decoder = clip.byte_decoder()

            prompttext = ''
            ids = []
            current_ids = []

            def dump(last=False):
                nonlocal prompttext, ids, current_ids
                if len(current_ids) == 1 and current_ids[0] < 0:
                    #special case word, negative of location in notreallytokens array
                    word = notreallytokens[-current_ids[0]] + " "
                else:
                    words = [vocab.get(x, "") for x in current_ids]
                    try:
                        word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode("utf-8")
                    except UnicodeDecodeError:
                        if last:
                            word = "❌" * len(current_ids)
                        elif len(current_ids) > 4:
                            id = current_ids[0]
                            ids += [id]
                            local_ids = current_ids[1:]
                            prompttext += "❌"

                            current_ids = []
                            for id in local_ids:
                                current_ids.append(id)
                                dump()
                                return
                        else:
                            return

                word = word.replace("</w>", " ")
                prompttext += word
                ids += current_ids
                current_ids = []

            for token in tokens:
                token = int(token)
                current_ids.append(token)
                dump()

            dump(last=True)
            return prompttext, ids

        # variable setup
        notreallytokens = ["0placeholder"]

        if p.seed == -1:
            p.seed = randint(1,4294967295)

        if neg_pos == "Positive":
            initial_prompt =  p.prompt
            prompt = p.prompt
        else:
            initial_prompt =  p.negative_prompt
            prompt = p.negative_prompt

        full_prompt, tokens = tokenize(prompt)

        # process the ignored items
        commonlist = [320, 539, 593, 518, 550] # a of with the an
        if ignorecommon:
            tokens = list(filter(lambda x: x not in commonlist, tokens))

        punctlist = [267,269,281]  # ,.:
        if ignorepunct:
            tokens = list(filter(lambda x: x not in punctlist, tokens))

        # handle specific limitations or requirements of chosen functionality

        if customword =="" or customword is None:
            customword = "blank"
        customword_id = addnotreallytoken(customword)

        if tokenrearrange == "Shuffle All" and take_x_atonce == 1:
            take_x_atonce = 2  # no point in shuffling if only one token at a time, so force to 2 minimum

        if tokenreweight == "Before and After":
            take_x_atonce = 2 # before and after means 2 tokens are needed.
            # if we add functions that change a single token into 2+, this will need to be revised

        # checked the negative box, TODO, see if I can get negative slider values
        if powerneg:
            power = -power

        # what if you do nothing...
        if tokenrearrange == "" and tokenreweight == "" and tokenreplacement == "" and tokenremoval == "":
            # let's default to removal of the items
            tokenremoval = "Remove Word/Token/Group"

        tokenslength = len(tokens)
        vocab_size = vocabsize()

        if take_x_atonce > tokenslength:
            take_x_atonce = tokenslength

        #startat and endat are 0-100 %, so we divide by 100, and then round result
        if startat >= endat: # don't do this, if so, ignore it entirely
            startat = 0
            endat = 100
        starttoken = round(tokenslength * startat / 100)  # so 0% = length * zero or token 0
        endtoken = round(tokenslength * endat / 100)  # so 100% = length * 1 or full tokenlength  (since range is always end-1, this is fine)

        p.do_not_save_samples = True

        # first generate the potentially 'cleaned' prompt (which may or may not be different from the original), and label, and setup things to add new images afterward
        clean_prompt,clean_token_ids = tokenize(tokens,True)
        print(f"\n{clean_prompt}")

        if neg_pos == "Positive":
            p.prompt = clean_prompt
        else:
            p.negative_prompt = clean_prompt

        proc = process_images(p)
        proc.images[0] = write_on_image(proc.images[0], "Cleaned Prompt")
        if shared.opts.samples_save:
            images.save_image(proc.images[0], p.outpath_samples, "", proc.seed, p.prompt, shared.opts.samples_format, info=proc.infotexts[0], p=p)

        # loop to process prompt changes, and generate an image for each prompt.

        for g in range(endtoken):  # loops from zero to 1 less than end token, ie every token 0 thru last desired token

            #items to reset each loop
            new_tokens = tokens.copy()

            if g < starttoken: # if we are skipping from the start
                continue
            if g > (endtoken - take_x_atonce):  # avoid grabbing last tokens if we're grouping, and not enough are left to grab a full set
                break

            #process the rest of the prompt, except for the piece under question:
            working_preprompt,returned_pretoken_ids = tokenize(new_tokens[:g],True)
            working_postprompt,returned_posttoken_ids = tokenize(new_tokens[g+take_x_atonce:],True)

            # get the tokens in question:
            working_tokens = new_tokens[g:g+take_x_atonce]
            original_prompt, working_tokens = tokenize(working_tokens,True)
            working_prompt = original_prompt

            # process all functions desired, this can now be more than one...

            # handle token removal functions, then token adding, then token replacement, then token reorganizing, then wrapping/weights
            # this order should make the most sense...

            # functions that reduce tokens to just one

            if tokenreplacement == "Change All to 1 Custom": # Replace group with just 1 custom word
                working_prompt,working_tokens = tokenize([customword_id],True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            if tokenreplacement == "Change All to 1 Random":  # Replace group Randomly with just 1
                working_tokens = [randint(1,vocab_size)]
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            # potentially changes token counts, either up or down...
            if tokenremoval == "Remove One or More of a Group": # Remove one or more of group
                #todo
                pass

            if tokenremoval == "Remove First in Group": # Remove first of group
                working_tokens.pop(0)
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            if tokenremoval == "Remove Last in Group": # Remove last of group
                working_tokens.pop()
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            # partial token replacement, leaves token count intact...
            if tokenreplacement == "Change First to Custom": # Replace first in group with custom word
                working_tokens[0] = customword_id
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            if tokenreplacement == "Change Last to Custom": # Replace last in group with custom word
                working_tokens[-1] = customword_id
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            if tokenreplacement == "Change First to Random": # Replace First Randomly in group
                random_token = randint(1,vocab_size)
                working_tokens[0] = random_token
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            if tokenreplacement == "Change Last to Random": # Replace Last Randomly in group
                random_token = randint(1,vocab_size)
                working_tokens[-1] = random_token
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            if tokenreplacement == "Change Each to Random": # Replace each Randomly in group
                for x in range(len(working_tokens)):
                    random_token = randint(1,vocab_size)
                    working_tokens[x] = random_token
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            # rearrange token order functions

            if tokenrearrange == "Shuffle All": # Shuffle group
                shufcount = 0
                original_tokens_order = working_tokens.copy()
                while shufcount < 5:  # if 5 mixes fail to find a different order than the input, move on... to handle poor shuffling of small amount of tokens
                    random.shuffle(working_tokens)
                    if working_tokens != original_tokens_order:
                        shufcount = 5
                    shufcount += 1
                working_prompt,working_tokens = tokenize(working_tokens,True)
                image_caption = f"{original_prompt}->\n{working_prompt}"

            # functions that add wrapping items like weights/etc

            if tokenreweight == "Before and After" and len(working_tokens) == 2: # Before and After
                token0,returned_token_id = tokenize([working_tokens[0]],True)
                token1,returned_token_id = tokenize([working_tokens[1]],True)
                working_prompt = "[" + token0.rstrip(" ") + ":" + token1.rstrip(" ") + ":" + str(power) + "] "
                image_caption = working_prompt
                # this really isn't conductive to further manipulation so it should be one of the last functions

            if tokenreweight == "Change Weight": # Strengthen or Weaken
                # do we already have a weight already we can change?
                if re.search(r":[0-9\.-]+", working_prompt):  # TODO do not changing bracketed weights, only parens or carets
                   # yes, let's change that one instead of wrapping it with a new one
                   working_prompt = re.sub(r":[0-9\.-]+", ':'+str(power), working_prompt)
                   # potentially this mean a group won't get a weight if one part is already weighted TODO fix
                else:
                   # no weight in current piece so wrap it all with a weight
                   working_prompt = f"({working_prompt.rstrip(' ')}:{power}) "
                # this really isn't very conductive to further manipulation so it should be one of the last functions
                image_caption = working_prompt

            # the nuclear option - remove entire section in question, no matter was done above, this will just wipe it out.
            if tokenremoval == "Remove Word/Token/Group": # Remove all of group
                image_caption = "No " + working_prompt
                working_prompt = ""
                # this really isn't conductive to any further manipulation so it should be the last function.  Boom.

            # put things back together, new_preprompt + working_prompt + new_postprompt
            new_prompt = working_preprompt + working_prompt + working_postprompt

            # this puts the prompt up before the generation, useful for console watching
            print(f"\n{new_prompt}")

            if neg_pos == "Positive":
                p.prompt = new_prompt
            else:
                p.negative_prompt = new_prompt

            appendimages = process_images(p)
            proc.images.append(appendimages.images[0])
            proc.infotexts.append(appendimages.infotexts[0])
            proc.images[-1] = write_on_image(proc.images[-1], image_caption)
            if shared.opts.samples_save:
                images.save_image(proc.images[-1], p.outpath_samples, "", proc.seed, p.prompt, shared.opts.samples_format, info=proc.infotexts[-1], p=p)
        # loop end

        # generate the original prompt as given, and label, this way the final result is the prompt is recorded as identical to the original given, and restore works correctly.
        # otherwise, the last prompt, no matter how mangled would be the recorded one, undesired.
        print(f"\n{initial_prompt}")

        if neg_pos == "Positive":
            p.prompt = initial_prompt
        else:
            p.negative_prompt = initial_prompt

        appendimages = process_images(p)
        proc.images.append(appendimages.images[0])
        proc.infotexts.append(appendimages.infotexts[0])
        proc.images[-1] = write_on_image(proc.images[-1], "Original Prompt")
        if shared.opts.samples_save:
            images.save_image(proc.images[-1], p.outpath_samples, "", proc.seed, p.prompt, shared.opts.samples_format, info=proc.infotexts[-1], p=p)

        # make grid, if desired, since all images are generated
        grid_flags = self.grid_options_mapping[grid_option]
        unwanted_grid_because_of_img_count = len(proc.images) < 2 and shared.opts.grid_only_if_multiple
        if ((shared.opts.return_grid or shared.opts.grid_save) and not p.do_not_save_grid and not grid_flags.never_grid and not unwanted_grid_because_of_img_count) or grid_flags.always_grid:
            grid = images.image_grid(proc.images)
            proc.images.insert(0,grid)
            proc.infotexts.insert(0, proc.infotexts[0])
            if shared.opts.grid_save or grid_flags.always_save_grid:
                images.save_image(grid, p.outpath_grids, "grid", p.seed, p.prompt, shared.opts.grid_format, info=proc.info, short_filename=not shared.opts.grid_extended_filename, p=p, grid=True)

        return proc
