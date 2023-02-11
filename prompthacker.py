# a merge of concepts/code from 'test my prompt' and 'tokenizer'
# to rip your prompt into small bits and then staple it back to together...
#
# authored by Scruffynerf  - msg me @Scruffy in Civitai discord

from modules.shared import opts, cmd_opts, state
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from PIL import Image, ImageFont, ImageDraw, ImageOps
from fonts.ttf import Roboto
import modules.scripts as scripts
from modules import script_callbacks, shared
import gradio as gr
from collections import namedtuple
from random import randint, sample
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer

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
        neg_pos = gr.Dropdown(label="Hack apart the negative or positive prompt", choices=["Positive","Negative"], value="Positive")
        hackfunction = gr.Dropdown(label="Function", choices=["Remove","Randomize","Shuffle","Strengthen or Weaken"], value="Remove")
        skip_x_first = gr.Slider(minimum=0, maximum=32, step=1, label='Skip X first tokens', value=0)
        take_x_atonce = gr.Slider(minimum=1, maximum=32, step=1, label='Take X tokens at a time', value=1)
        power = gr.Slider(minimum=-2, maximum=2, step=0.1, label='Stronger/Weaker value', value=1)
        grid_option = gr.Radio(choices=list(self.grid_options_mapping.keys()), label='Grid generation', value=self.default_grid_opt)
        font_size = gr.Slider(minimum=12, maximum=64, step=1, label='Font size', value=32)
        return [neg_pos,hackfunction,skip_x_first,take_x_atonce,power,grid_option,font_size]

    def run(self,p,neg_pos,hackfunction,skip_x_first,take_x_atonce,power,grid_option,font_size):
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

        def tokenShuffle(x, *s):
            x[slice(*s)] = sample(x[slice(*s)], len(x[slice(*s)]))


        def tokenize(promptinput, input_is_ids=False):
            clip = shared.sd_model.cond_stage_model.wrapped
            if isinstance(clip, FrozenCLIPEmbedder):
                clip = VanillaClip(shared.sd_model.cond_stage_model.wrapped)
            elif isinstance(clip, FrozenOpenCLIPEmbedder):
                clip = OpenClip(shared.sd_model.cond_stage_model.wrapped)
            else:
                raise RuntimeError(f'Unknown CLIP model: {type(clip).__name__}')

            if input_is_ids:
                tokens = promptinput
            else:
                tokens = shared.sd_model.cond_stage_model.tokenize([promptinput])[0]

            vocab = {v: k for k, v in clip.vocab().items()}
            prompttext = ''
            ids = []
            current_ids = []
            class_index = 0

            byte_decoder = clip.byte_decoder()

            def dump(last=False):
                nonlocal prompttext, ids, current_ids
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

        p.do_not_save_samples = True
        initial_seed = p.seed
        vocab_size = vocabsize()
        if initial_seed == -1:
            initial_seed = randint(1,9999999)
        if neg_pos == "Positive":
            initial_prompt =  p.prompt
            prompt = p.prompt
        else:
            initial_prompt =  p.negative_prompt
            prompt = p.negative_prompt

        full_prompt, tokens = tokenize(prompt)
        print("total images :", len(tokens))
        first = True

        if hackfunction == "Shuffle" and take_x_atonce == 1:
           take_x_atonce = 2

        for g in reversed(range(len(tokens) +2 -take_x_atonce )):
            new_prompt = ""
            new_prompt2 = ""
            new_tokens = []
            print(f"g={g}")
            f = g-1
            if f >= 0 and f < skip_x_first:
                continue
            if f >= 0 and f > (len(tokens) - take_x_atonce):
                continue
            if f >= 0:
                if hackfunction == "Remove":
                    new_tokens = tokens[:f] + tokens[f+take_x_atonce:]
                    new_prompt,returned_token_ids = tokenize(new_tokens,True)
                    removed_tokens = tokens[f:f+take_x_atonce]
                    image_caption,returned_removed_tokens = tokenize(removed_tokens,True)
                    image_caption = "No: " + image_caption
                elif hackfunction == "Randomize":
                    new_tokens = tokens.copy()
                    flimit = f+take_x_atonce
                    if flimit > len(new_tokens)-1:
                       flimit = len(new_tokens)-1
                    for x in range(f,flimit+1):
                            random_token = str(randint(1,vocab_size))
                            print(f"random:{random_token}")
                            new_tokens[x] = random_token
                    random_tokens = new_tokens[f:flimit]
                    newtext,returned_new_tokens = tokenize(random_tokens,True)
                    removed_tokens = tokens[f:flimit]
                    oldtext,returned_removed_tokens = tokenize(removed_tokens,True)
                    image_caption = f"{oldtext}->\n{newtext}"
                    new_prompt,returned_token_ids = tokenize(new_tokens,True)
                elif hackfunction == "Shuffle":
                    new_tokens = tokens.copy()
                    shufcount = 0
                    original_ordered_tokens = tokens[f+1:f+take_x_atonce+1]
                    while shufcount < 5:
                        tokenShuffle(new_tokens,f+1,f+take_x_atonce+1)
                        reordered_tokens = new_tokens[f+1:f+take_x_atonce+1]
                        if reordered_tokens == original_ordered_tokens:
                           shufcount += 1
                        else:
                           shufcount = 5
                    newtext,returned_new_tokens = tokenize(reordered_tokens,True)
                    removed_tokens = tokens[f+1:f+take_x_atonce+1]
                    oldtext,returned_removed_tokens = tokenize(removed_tokens,True)
                    image_caption = f"{oldtext}->\n{newtext}"
                    new_prompt,returned_token_ids = tokenize(new_tokens,True)
                elif hackfunction == "Strengthen or Weaken":
                    new_tokens = tokens.copy()
                    change_tokens = new_tokens[f:f+take_x_atonce]
                    newtext,returned_new_tokens = tokenize(change_tokens,True)
                    image_caption = f" ({newtext.rstrip(' ')}:{power}) "
                    #new_tokens[f:f+take_x_atonce] = returned_new_tokens
                    new_prompt1,returned_token_ids = tokenize(new_tokens[:f],True)
                    new_prompt2,returned_token_ids = tokenize(new_tokens[f+take_x_atonce:],True)
                    new_prompt = new_prompt1 + image_caption + new_prompt2
                #              new_prompt += image_caption + " "
                #           new_tokens2 = new_tokens2 + str(tokens[x]) + ","
                #        else:
                #           if x == f and new_tokens != "":
                #              new_prompt,starting_tokens = tokenize(new_tokens.rstrip(","),True)
                #              new_prompt += " "
                #           original_token += str(tokens[x]) + ","
                #    if new_tokens2 != "":
                #        new_prompt2,ending_tokens = tokenize(new_tokens2.rstrip(","),True)
                #        new_prompt += new_prompt2
                #    else:
                #        originaltext,original_tokens = tokenize(original_token,True)
                #        image_caption += f"({originaltext.rstrip(' ')}:{power})"
                #~        new_prompt += image_caption
            else:
                new_prompt = initial_prompt
            print(f"{new_prompt}")
            if neg_pos == "Positive":
                p.prompt = new_prompt
            else:
                p.negative_prompt = new_prompt
            p.seed = initial_seed
            if first:
                proc = process_images(p)
                first = False
            else:
                appendimages = process_images(p)
                proc.images.insert(0,appendimages.images[0])
                proc.infotexts.insert(0,appendimages.infotexts[0])
            if f >= 0:
                proc.images[0] = write_on_image(proc.images[0], image_caption)
            else:
                proc.images[0] = write_on_image(proc.images[0], "Original Prompt")

            if opts.samples_save:
                images.save_image(proc.images[0], p.outpath_samples, "", proc.seed, p.prompt, opts.samples_format, info= proc.info, p=p)

        grid_flags = self.grid_options_mapping[grid_option]
        unwanted_grid_because_of_img_count = len(proc.images) < 2 and opts.grid_only_if_multiple
        if ((opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not grid_flags.never_grid and not unwanted_grid_because_of_img_count) or grid_flags.always_grid:
            grid = images.image_grid(proc.images)
            proc.images.insert(0,grid)
            proc.infotexts.insert(0, proc.infotexts[-1])
            if opts.grid_save or grid_flags.always_save_grid:
                images.save_image(grid, p.outpath_grids, "grid", initial_seed, initial_prompt, opts.grid_format, info=proc.info, short_filename=not opts.grid_extended_filename, p=p, grid=True)
        return proc

