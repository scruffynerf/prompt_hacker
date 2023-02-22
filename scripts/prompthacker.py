# Inspired by (and started out using concepts/code from) 'Test My Prompt' and 'Tokenizer'
# But then it was zapped by lighting and it's alive!!!!  Frau Blucher!  NEEEEIIGGHHHH!
# Purpose: to rip your precious prompt into small bits and then
# we'll staple/tape/pound it back to together in new and interesting ways...

# WARNING MASSIVELY BETA CODE with a major rewrites... still debugging and refactoring.

# Authored by Scruffynerf  - msg me @Scruffy in the Civitai Discord

import random
import re
import requests
from collections import namedtuple
import modules.scripts as scripts
import modules.prompt_parser as prompt_parser
from modules.processing import Processed, StableDiffusionProcessingImg2Img, process_images, images
from modules import script_callbacks, shared, sd_hijack
from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder
import open_clip.tokenizer
from PIL import Image, ImageFont, ImageDraw, ImageOps
from fonts.ttf import Roboto
import gradio as gr
from textwrap import wrap
import os
import difflib

# datamuse api code from https://github.com/gmarmstrong/python-datamuse
resmin = 5  # minimum results we want

class Datamuse(object):

    def __init__(self, max_results=20):
        self.api_root = 'https://api.datamuse.com'
        self._validate_max(max_results)
        self.max = max_results

    def __repr__(self):
        return '\n'.join(
                    ['{0}: {1}'.format(k, v) for k, v in requests.api.__dict__.items()])

    @staticmethod
    def _validate_max(max_results):
        if not(0 < max_results <= 1000):
            raise ValueError("Datamuse only supports values of max in (0, 1000]")

    def _get_resource(self, endpoint, **kwargs):
        url = '/'.join([self.api_root, endpoint])
        response = requests.get(url, params=kwargs)
        #print(f"{response.json()}")  # debug
        return response.json()

    def set_max_default(self, max_results):
        self._validate_max(max_results)
        self.max = max_results

    def words(self, **kwargs):
        """
        This endpoint returns a list of words (and multiword expressions) from
        a given vocabulary that match a given set of constraints.
        See <https://www.datamuse.com/api/> for the official Datamuse API
        documentation for the `/words` endpoint.
        :param `**kwargs`: Query parameters of constraints and hints.
        :return: A list of words matching that match the given constraints.
        """
        if 'max' not in kwargs:
            kwargs.update({'max': self.max})
        return self._get_resource('words', **kwargs)

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

actionchoices = ["",
                "Remove",
                "Replace",
                "Add Before",
                "Add After",
                "Shuffle entire group"]
                
                #"Change Weight to V1(+V2+V3)",
                #"Change Weight(s) to Random 0-1",
                #"Before+After [A:B:V1] (min Group of 2)",
                #"Fusion Att. Interpolate (A:V1,V2)",
                #"Fusion Linear Interpolate [A,B,C+:V1,V2,V3]",
                #"Fusion Catmull Interpolate [A,B,C+:V1,V2,V3]",
                #"Fusion Bezier Interpolate [A,B,C+:V1,V2,V3]",
                #"Fusion [AddA:WhenV1]",
                #"Fusion [RemoveA::WhenV1]",
                #"Fusion [A:B:StartV1,StopV2]"]

withchoices =  ["",
                "Random",
                "Custom",
                "Rhymes with X",
                "Meaning of X",
                "Sounds like X",
                "Spelled like X",
                "Related to X",
                "Synonym of X",
                "Antonym of X",
                "Specific X",
                "Generic X",
                "Part of X",
                "X Is Part of",
                "Adjective<->Noun"]
                # TODO "X Follows/Precedes X","Topics/Hint/ContextLeft/Right",
                # "Photography Terms","Artist Names", "Artistic Styles",
                #  "MetroArtMuseum Random Artist"]
                
whichchoices = ["",
                "Group as One",
                "First in Group",
                "Once in Group",
                "Randomly in Group",
                "Last in Group",
                "Every in Group"]


# TODO add support for PIPE  [A|B] will swap



class Script(scripts.Script):
    GridSaveFlags = namedtuple('GridSaveFlags',
                                ['never_grid', 'always_grid', 'always_save_grid'],
                                defaults=(False, False, False))
    grid_options_mapping = {
        "Use user settings": GridSaveFlags(),
        "Don't generate": GridSaveFlags(never_grid=True),
        "Generate": GridSaveFlags(always_grid=True),
        "Generate and always save": GridSaveFlags(always_grid=True,
            always_save_grid=True),
        }
    default_grid_opt = list(grid_options_mapping.keys())[-1]

    def title(self):
        return "Prompt Hacker"

    def ui(self, is_img2img):

        def enable_hack1b(choice):
            vis = (choice in ["Remove", "Replace", "Add Before", "Add After"])
            return [gr.Dropdown.update(value="")]

    #with gr.Blocks():
        #    s = gr.Slider(1, max_textboxes, value=max_textboxes, step=1, label="How many textboxes?")
        #for i in range(max_textboxes):
        #    t = gr.Textbox(f"Textbox {i}")
        #    atextboxes.append(t)
        #    t1 = gr.Dropdown(label="Action", choices=actionchoices,value=["Remove","Replace"],multiselect=True,Interactive=True)
        #    t2 = gr.Textbox(f"Textbox 2")
        #    t3 = gr.Textbox(f"Textbox 3")
        #    t4 = gr.Textbox(f"Textbox 4")
        #    t5 = gr.Textbox(f"Textbox 5")
        #    s.change(variable_outputs, s, [t1, t2, t3, t4, t5])

        with gr.Blocks():
            with gr.Row():
                neg_pos = gr.Dropdown(label="Hack the positive or negative prompt?",
                choices=["Positive","Negative"], value="Positive")
            with gr.Row():
                startat = gr.Slider(minimum=0, maximum=100, step=1,
                    label='Start at % of prompt', value=0)
                endat = gr.Slider(minimum=0, maximum=100, step=1,
                    label='Stop at % of prompt', value=100)
                take_x_atonce = gr.Slider(minimum=1, maximum=100, step=1,
                     label='Group X words/tokens at a time', value=1)
            with gr.Accordion(label="Misc Settings",open=False):
                nosplitwords = gr.Checkbox(
                        label='Keep compound tokens (vs split into 2+ tokens)',
                        value=True)
                magicwords = gr.Textbox(
                        label="Special Word handling (if not autorecognized)",
                        lines=1, value="yourcustomword,yourcustomword2")
                caption_preference = gr.Dropdown(label="Your Caption Preference",
                    choices=[ "visual diff",
                            "old part->new part",
                            "new part only",
                            "entire new prompt",
                            "no caption at all"], value="visual diff")
                font_size = gr.Slider(
                        minimum=12, maximum=64, step=1, label='Font size', value=32)
                with gr.Box():
                    grid_option = gr.Radio(
                        choices=list(self.grid_options_mapping.keys()),
                        label='Grid generation', value=self.default_grid_opt)
                ignorecommon = gr.Textbox(
                        label="Remove these (comma separated) words from my prompt",
                        lines=1, value="a,an,the,I,i,me,as,is")
                ignorepunct = gr.Textbox(
                        label="Remove these characters from my prompt entirely",
                        lines=1, value="_")
            with gr.Accordion(label="Hack 1",open=False):
                with gr.Box():
                    with gr.Row():
                        hack1a = gr.Dropdown(label="Action",choices=actionchoices,value="")
                        hack1b = gr.Dropdown(label="Which",choices=whichchoices,value="")
                        hack1c = gr.Dropdown(label="With",choices=withchoices,value="")
                    with gr.Row():
                        hack1g = gr.Textbox(label="Custom word(s) go here", lines=1,value="")
                    with gr.Row():
                        hack1d = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V1', value=0)
                        hack1e = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V2', value=0)
                        hack1f = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V3', value=0)
                        
                   
            with gr.Accordion(label="Hack 2",open=False):
                with gr.Box():
                    with gr.Row():
                        hack2a = gr.Dropdown(label="Action",choices=actionchoices,value="")
                        hack2b = gr.Dropdown(label="Which",choices=whichchoices,value="")
                        hack2c = gr.Dropdown(label="With",choices=withchoices,value="")
                    with gr.Row():
                        hack2g = gr.Textbox(label="Custom word(s) go here", lines=1,value="")
                    with gr.Row():
                        hack2d = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V1', value=0)
                        hack2e = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V2', value=0)
                        hack2f = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V3', value=0)
                        
            with gr.Accordion(label="Hack 3",open=False):
                with gr.Box():
                    with gr.Row():
                        hack3a = gr.Dropdown(label="Action",choices=actionchoices,value="")
                        hack3b = gr.Dropdown(label="Which",choices=whichchoices,value="")
                        hack3c = gr.Dropdown(label="With",choices=withchoices,value="")
                    with gr.Row():
                        hack3g = gr.Textbox(label="Custom word(s) go here", lines=1,value="")
                    with gr.Row():
                        hack3d = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V1', value=0)
                        hack3e = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V2', value=0)
                        hack3f = gr.Slider(minimum=-10.0, maximum=10, step=0.1, label='V3', value=0)
            
            
        return [neg_pos, startat, endat, take_x_atonce, 
                nosplitwords, ignorepunct, ignorecommon, magicwords,
                font_size, grid_option, caption_preference,
                hack1a, hack1b, hack1c, hack1d, hack1e, hack1f, hack1g,
                hack2a, hack2b, hack2c, hack2d, hack2e, hack2f, hack2g,
                hack3a, hack3b, hack3c, hack3d, hack3e, hack3f, hack3g]

    def run(self, p, neg_pos, startat, endat, take_x_atonce, 
                  nosplitwords, ignorepunct, ignorecommon, magicwords,
                  font_size, grid_option, caption_preference,
                  hack1a, hack1b, hack1c, hack1d, hack1e, hack1f, hack1g,
                  hack2a, hack2b, hack2c, hack2d, hack2e, hack2f, hack2g,
                  hack3a, hack3b, hack3c, hack3d, hack3e, hack3f, hack3g):

        def write_on_image(img, msg):
            ix,iy = img.size
            draw = ImageDraw.Draw(img)
            margin = 1
            height = 2
            charsperline = 25
            font_size = 40
            fontsize = font_size
            font_path = scripts.basedir() + '/extensions/prompt_hacker/' + 'spacemono.ttf'
            font = ImageFont.truetype(font_path, fontsize)
            origmsg = msg
                            
            if msg not in ["Original Prompt","Cleaned Prompt"]:              
                #first pass attempt as directed
                textlist = wrap( msg, int(charsperline*32/fontsize), drop_whitespace=True, break_on_hyphens=True)
                msg = "\n".join(textlist)
                tx = draw.textbbox((0,0),msg,font,spacing=3, stroke_width=2)
                
                while (tx[2]>ix) or (tx[3]>(iy*.4)) or fontsize <= 4:
                    # too much of the image will be blocked by text, so lower fontsize and retry
                    fontsize = int(fontsize * .95)
                    if fontsize < 4:
                        #whoops, this is a problem
                        fontsize = 4
                    textlist = wrap(origmsg, int(charsperline*32/fontsize), drop_whitespace=True, break_on_hyphens=True)
                    msg = "\n".join(textlist)
                    font = ImageFont.truetype(font_path, fontsize)
                    tx = draw.textbbox((0,0),msg,font,spacing=3, stroke_width=2)

            tx = draw.textbbox((0,0),msg,font,spacing=3, stroke_width=2)
            #draw.text((int((ix-tx[2])/2),text_height+margin),msg,(0,0,0),font=font)
            #draw.text((int((ix-tx[2])/2),text_height-margin),msg,(0,0,0),font=font)
            #draw.text((int((ix-tx[2])/2+margin),text_height),msg,(0,0,0),font=font)
            #draw.text((int((ix-tx[2])/2-margin),text_height),msg,(0,0,0),font=font)
            draw.text((int((ix-tx[2])/2),iy-tx[3]-10), msg,(255,255,255),font=font, spacing=3, stroke_width=2, stroke_fill=(0,0,0))
            return img
            
        def promptdiffimg(img, originalprompt, newprompt):
            
            #print(f"clean prompt:  {originalprompt}")
            #print(f"new prompt:  {newprompt}")
            ix,iy = img.size
            draw = ImageDraw.Draw(img)
            margin = 1
            height = 2
            charsperline = 25
            linespacing = 3
            font_size = 48
            fontsize = font_size
            font_path = scripts.basedir() + '/extensions/prompt_hacker/' + 'spacemono.ttf'
            font = ImageFont.truetype(font_path, fontsize)

            d = difflib.Differ()
            
            origp = re.split(r'([^A-Za-z0-9])', originalprompt)
            newp = re.split(r'([^A-Za-z0-9])', newprompt)
            difflist = d.compare(origp, newp)

            same = []
            old = []
            new = []
            everything = []

            for word in difflist:
                #print(f"evaling {word}")
                typeofword = word[:2]
                thisword = word[2:]
                wordsize = " " * len(thisword)                
                if typeofword == "- ":
                    # old
                    old.append(thisword)
                    everything.append(thisword)
                    new.append(wordsize)
                    same.append(wordsize)
                elif typeofword == "+ ":
                    # new
                    new.append(thisword)
                    everything.append(thisword)
                    old.append(wordsize)
                    same.append(wordsize)
                elif typeofword == "  ":
                    same.append(thisword)
                    everything.append(thisword)
                    new.append(wordsize)
                    old.append(wordsize)

            #split into different texts, with spaces for the other two so all 3 are same size
            sametext = "".join(same)
            oldtext = "".join(old)
            newtext = "".join(new)
            everythingtext = "".join(everything)

            #print(everythingtext)
            #print(sametext)
            #print(oldtext)
            #print(newtext)

            #first pass attempt as directed
            textlist = wrap(everythingtext, int(charsperline*32/fontsize), drop_whitespace=False, break_on_hyphens=False)
            msg = "\n".join(textlist)
            tx = draw.textbbox((0,0),msg,font,spacing=linespacing, stroke_width=2)
            tryagain = True
            while ((tx[2]>ix) or (tx[3]>(iy*.4))) and tryagain:
                    # too much of the image will be blocked by text, so lower fontsize and retry
                    fontsize = int(fontsize * .95)
                    if fontsize < 4:
                        #whoops, this is a problem, no smaller, sorry.
                        fontsize = 4
                        tryagain = False
                    linespacing = int(3.99 * fontsize/64)+1
                    textlist = wrap(everythingtext, int(charsperline*32/fontsize), drop_whitespace=False, break_on_hyphens=False)
                    msg = "\n".join(textlist)
                    font = ImageFont.truetype(font_path, fontsize)
                    tx = draw.textbbox((0,0),msg,font,spacing=linespacing, stroke_width=2)

            #print(f"fontsize: {fontsize}")
            #print(f"textlist: {textlist}")

            count = 0
            oldsplit = []
            newsplit = []
            samesplit = []
            for item in textlist:
              size = len(item)
              oldsplit.append(oldtext[count:count+size])
              samesplit.append(sametext[count:count+size])
              newsplit.append(newtext[count:count+size])
              count = count + size
            samemsg = "\n".join(samesplit)
            oldmsg = "\n".join(oldsplit)
            newmsg = "\n".join(newsplit)
            
            msglist = []
            #msglist.append([msg,(255,255,255)])
            msglist.append([samemsg,(255,255,255)])
            msglist.append([oldmsg,(240,0,0)])
            msglist.append([newmsg,(0,255,0)])
            
            featurelist = ["-liga"]

            for msgdraw in msglist:
                draw.text((int((ix-tx[2])/2),iy-tx[3]-10),
                           msgdraw[0],msgdraw[1],
                           font=font, spacing=linespacing, features=featurelist, 
                           stroke_width=2, stroke_fill=(0,0,0))
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

        #def parseweight(word):
        #if re.match(r"(?P<start>[\[\(]+)(?P<word1>[^0-9\,\:]*?)?(?P<split1>[:,])?(?P<word2>[^0-9\,\:]*?)?(?P<split2>[:,])?(?P<word3>[^0-9\,\:]*?)?(?P<split3>[:,])?(?P<number1>[0-9\.-]+)?(?P<split4>[:,])?(?P<number2>[0-9\.-]+)?(?P<split5>[:,])?(?P<number3>[0-9\.-]+)?(?P<curve>:[([a-z]{6,7})?(?P<end>[\]\)]+)", word)

        def splitpromptinwordsandweights(text):
            mywords = []
            myweights = []
            commaonend = False
            itemcommaonend = False
            
            pair_results = prompt_parser.parse_prompt_attention(text)
            
            for index, pair in enumerate(pair_results):
                #print(index, pair)
                word = pair[0]
                if word.rstrip(",") != word and word != ",":
                    # comma on the end
                    commaonend = True
                    word = word.rstrip(",")

                weight = pair[1]
                if round(weight,2) == .91:
                    #whoops, no brackets so we can support other formats, so restore/reverse
                    mywords.append("["+word+"]")
                    myweights.append(1)
                elif word == " " and weight == 1:
                    continue # skipping not weighted spaces... we might want to make this configurable TODO
                elif word.count(' '): # we have a space...
                    splitwords = word.split(" ")
                    for index, item in enumerate(splitwords):
                        if item.rstrip(",") != item and item != ",":
                            # comma on the end
                            itemcommaonend = True
                            item = item.rstrip(",")
                        mywords.append(item)
                        myweights.append(weight)
                        if itemcommaonend:
                            mywords.append(",")
                            myweights.append(weight)
                            itemcommaonend = False
                        if index < len(splitwords)-1:
                            mywords.append(" ")
                            myweights.append(weight)
                else:
                    mywords.append(word)
                    myweights.append(weight)
                if commaonend:
                    mywords.append(",")
                    myweights.append(weight)
                    commaonend = False
            return mywords,myweights
            
        def combinewordsandweightsintoprompt(words,weights):
            #this isn't perfect, but it's more workable

            prompt = ""
            wordstack = ""

            for index, myword in enumerate(words):
                myweight = weights[index]
                #print(f"{index}: {myword}, {myweight}")
                if index != len(words)-1:
                    # not at the end should we combine?
                    if weights[index+1] == myweight:
                        # yes, combine
                        wordstack = wordstack + myword
                        continue                        
                if myweight == 1:
                    prompt += wordstack + myword
                elif round(myweight,2) == .91:
                    prompt += "[" + wordstack + myword + "]"
                elif round(myweight,1) == 1.1:
                    prompt += "(" + wordstack + myword + ")"
                else:
                    prompt += "(" + wordstack + myword + ":" + str(round(myweight,2)) + ")"
                #reset wordstack
                wordstack = ""                
            return prompt

        def tokenize(promptinput, input_is_ids=False):
            tokens = []
            # handle the empty case if this happens (it shouldn't, but...)
            if promptinput == "" or promptinput is [] or promptinput is None:
                return "",[],[]

            if input_is_ids:
                # if we've already got token ids, use em instead of breaking things down...
                tokens = promptinput
            else:
                # Just trusting the tokenizer code is no good,
                # as it doesn't support embeddings, loras, parens/brackets/weights
                # and some trigger words.
                splitintowords = promptinput.split(" ")
                for currentword in splitintowords:
                    thisword = currentword
                    commaonend = False
                    if thisword.rstrip(",") != thisword and thisword != ",":
                        # comma on the end
                        commaonend = True
                        thisword = thisword.rstrip(",")
                    if thisword in sd_hijack.model_hijack.embedding_db.word_embeddings.keys():
                        # this word is embedding
                        tokens.append(addnotreallytoken(thisword))
                    elif re.match(r"[<\(\[]", thisword):
                        # this word is a lora
                        # OR this word is weighted
                        # OR this word is weighted/interpolated
                        tokens.append(addnotreallytoken(thisword))
                    elif re.search(r"[0-9]", thisword):
                        # this 'word' has numbers, which SD tokenizer will split apart otherwise, so save it whole
                        tokens.append(addnotreallytoken(thisword))
                    elif thisword in magicwords.split(","):
                        # this word was given to us to flag specifically
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
                    # special case word, negative of location in notreallytokens array
                    word = notreallytokens[-current_ids[0]] + " "
                    if word[-2] == "*" and word[-3] == "*":
                        # handle special case, no space at end, remove "** "
                        word = word.rstrip("* ")
                else:
                    words = [vocab.get(x, "") for x in current_ids]
                    try:
                        word = bytearray([byte_decoder[x] for x in ''.join(words)]).decode("utf-8")
                    except UnicodeDecodeError:
                        if last:
                            word = "❌" * len(current_ids)
                        elif len(current_ids) > 4:
                            thisid = current_ids[0]
                            ids += [thisid]
                            local_ids = current_ids[1:]
                            prompttext += "❌"

                            current_ids = []
                            for thisid in local_ids:
                                current_ids.append(thisid)
                                dump()
                                return
                        else:
                            return

                word = word.replace("</w>", " ")
                #print(f"tokenizer debugging *{word}*")
                if word == ", " or word == ",":
                    #if word == ",":
                    #    word = word + " "
                    prompttext = prompttext.rstrip(" ")                    
                prompttext += word
                ids += current_ids
                current_ids = []

            for token in tokens:
                current_ids.append(int(token))
                dump()
            dump(last=True)
            
            prompttext = re.sub(r',', ', ', prompttext).strip(",")
            prompttext = re.sub(r',+', ',', prompttext)
            prompttext = re.sub(r' +', ' ', prompttext).strip()
            
            return prompttext, ids
            
        def randomword():
            random_start = chr(random.randrange(97, 97 + 26))+"*"
            random_word = random.choice(wordapi.words(sp=random_start))
            return random_word["word"]
            
        def rhymeswith(passed_word):
            returnedwords = wordapi.words(rel_rhy=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(rel_nry=passed_word))
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we STILL got nothing... orange orange boborange TBD
            return addnotreallytoken("rhyme")
        
        def meaningof(passed_word):
            returnedwords = wordapi.words(ml=passed_word)
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            # when in doubt, go with the classic meaning
            return addnotreallytoken("forty-two")

        def soundslike(passed_word):
            returnedwords = wordapi.words(sl=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(rel_hom=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... sounds like...
            return addnotreallytoken(passed_word+"ish")
 
        def spelledlike(passed_word):
            returnedwords = wordapi.words(sl=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(rel_cns=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... spelled like...
            return addnotreallytoken(passed_word+"(sic)")

        def relatedto(passed_word):
            returnedwords = wordapi.words(rel_trg=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(sl=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... reminds me of...
            return addnotreallytoken(passed_word+"punk")

        def synonym(passed_word):
            returnedwords = wordapi.words(rel_syn=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... reminds me of...
            return addnotreallytoken(passed_word+"-like")

        def antonym(passed_word):
            returnedwords = wordapi.words(rel_ant=passed_word)
            if len(returnedwords) < resmin:
                notword = "not "+passed_word
                returnedwords.extend(wordapi.words(ml=notword))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... opposite...
            return addnotreallytoken("("+passed_word+":-1)")

        def generic(passed_word):
            returnedwords = wordapi.words(rel_spc=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... science it...
            return addnotreallytoken(passed_word+"oid")

        def specific(passed_word):
            returnedwords = wordapi.words(rel_gen=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... organize it...
            return addnotreallytoken(passed_word+"ism")

        def partof(passed_word):
            returnedwords = wordapi.words(rel_par=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... detail it...
            return addnotreallytoken("detailed " + passed_word)

        def ispartof(passed_word):
            returnedwords = wordapi.words(rel_par=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... detail it...
            return addnotreallytoken("detailed " + passed_word)

        def partof(passed_word):
            returnedwords = wordapi.words(rel_com=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... style it...
            return addnotreallytoken(passed_word+ " style")

        def partof(passed_word):
            returnedwords = wordapi.words(rel_com=passed_word)
            if len(returnedwords) < resmin:
                returnedwords.extend(wordapi.words(ml=passed_word))
            if len(returnedwords):
                # pick one at random
                chosen = random.choice(returnedwords)
                return addnotreallytoken(chosen["word"])
            #we got nothing... style it...
            return addnotreallytoken(passed_word+ " style")

        def adjnoun(passed_word):
            #is this a noun or a adjective or neither?
            worddata = wordapi.words(sp=passed_word,md=p)
            if len(worddata):
                wordtype =  worddata[0]["tags"]
                if "n" in wordtype or "N" in wordtype:
                    #it's a noun or proper noun
                    returnedwords = wordapi.words(rel_jjb=passed_word)
                    if len(returnedwords) < resmin:
                        returnedwords.extend(wordapi.words(rel_jja=passed_word))
                    if len(returnedwords) < resmin:
                        returnedwords.extend(wordapi.words(ml=passed_word))
                    if len(returnedwords):
                        # pick one at random
                        chosen = random.choice(returnedwords)
                        return addnotreallytoken(chosen["word"])
                    else:
                        #we got nothing... focus it
                        return addnotreallytoken(passed_word+"focus")
                elif "adj" in wordtype or "adv" in wordtype:
                    #it's a adjective/adverb
                    returnedwords = wordapi.words(rel_jja=passed_word)
                    if len(returnedwords) < resmin:
                        returnedwords.extend(wordapi.words(rel_jjb=passed_word))
                    if len(returnedwords):
                        # pick one at random
                        chosen = random.choice(returnedwords)
                        return addnotreallytoken(chosen["word"])
                    else:
                        #we got nothing... more of it
                        return addnotreallytoken("extra "+passed_word)
                else:
                    #neither so...maybe a verb... add LY because Tom Lehrer said so.
                    return addnotreallytoken(passed_word.strip()+"ly")
            else: # no result? likely GIGO
                return random.randint(1,vocabsize())
                
        def pickwith(pickmethod, passed_token, custom=""):
            if type(passed_token) is list:
                # we were passed more than one token...
                # for now, process each
                # TODO? call this function for each OR only one at random
                for i in range(len(passed_token)):
                    passed_token[i] = pickwith(pickmethod, passed_token[i], custom)
                return passed_token
            else:
                if passed_token == 267 and pickmethod not in ["Custom", "Random"]:
                    #comma, don't bother
                    return passed_token
                # we were passed a single token, so let's get the related word(s)
                unclean_word,passed_token = tokenize([passed_token],True)
                # is this a complex item, with a weight?
                #if re.search(r":[0-9\.-]+", unclean_word): TBD TODO
                # handle commas

                if unclean_word != unclean_word.rstrip(" "):
                    tokenisnotfullword = False
                else:
                    #token _lacked_ space at end, meaning it's likely a partial word...
                    tokenisnotfullword = True
                    #TBD how to handle this...
                unclean_word = unclean_word.strip()
                # we should probably strip of all nonA-Z for use with api
                # might want to parse and save weights/etc TODO
                passed_word = re.sub(r'[^a-zA-Z ]', '', unclean_word).strip()
                if passed_word == "":
                    passed_word = randomword()
            #
            # figure out which method we're using
            #
            if pickmethod == "Custom": 
                if custom == "":
                    # you didn't give a custom word, but requested Custom, so we're going to use the random method instead
                    picked_token = random.randint(1,vocabsize())
                else:
                    picked_token = custom
            #        
            if pickmethod == "Random":
                picked_token = random.randint(1,vocabsize())
                #
            elif pickmethod == "Rhymes with X":
                picked_token = rhymeswith(passed_word)
                #
            elif pickmethod == "Meaning of X":
                picked_token = meaningof(passed_word)
                #
            elif pickmethod == "Sounds like X":
                picked_token = soundslike(passed_word)
                #
            elif pickmethod == "Spelled like X":
                picked_token = spelledlike(passed_word)
                #
            elif pickmethod == "Related to X":
                picked_token = relatedto(passed_word)
                #
            elif pickmethod == "Synonym of X":
                picked_token = synonym(passed_word)
                #
            elif pickmethod == "Antonym of X":
                picked_token = antonym(passed_word)
                #
            elif pickmethod == "Generic X":
                picked_token = generic(passed_word)
                #
            elif pickmethod == "Specific X":
                picked_token = specific(passed_word)
                #
            elif pickmethod == "X is Part of":
                picked_token = ispartof(passed_word)
                #
            elif pickmethod == "Part of X":
                picked_token = partof(passed_word)
                #
            elif pickmethod == "Adjective<->Noun":
                picked_token = adjnoun(passed_word)
                #
            #elif pickmethod == "":
            #    picked_token = random.randint(1,vocabsize())
            #
            #print(f"{pickmethod}: {passed_token} -> {picked_token}")
            return picked_token

        # main variable setup
        notreallytokens = ["0placeholder"]
        wordapi = Datamuse()
        wordweights = []
        p.do_not_save_samples = True

        # workaround for lack of array handling in UI? TODO fix if possible
        hacks = []
        hack1 = {"order":1, "action": hack1a, "which": hack1b,"with": hack1c,
            "value1":hack1d,  "value2":hack1e, "value3":hack1f, "custom": hack1g }
        hacks.append(hack1)
        hack2 = {"order":2, "action": hack2a, "which": hack2b,"with": hack2c,
            "value1":hack2d,  "value2":hack2e, "value3":hack2f, "custom": hack2g }
        hacks.append(hack2)
        hack3 = {"order":3, "action": hack3a, "which": hack3b,"with": hack3c,
            "value1":hack3d,  "value2":hack3e, "value3":hack3f, "custom": hack3g }
        hacks.append(hack3)

        if p.seed == -1:
            p.seed = random.randint(1,4294967295)

        if neg_pos == "Positive":
            initial_prompt =  p.prompt
            prompt = p.prompt.strip()
        else:
            initial_prompt =  p.negative_prompt
            prompt = p.negative_prompt.strip()

        mywords,myweights = splitpromptinwordsandweights(prompt)
        
        # Changed, let's filter out common words and punctuation _before_ we tokenize.  
        # Tokens vary too much to parse, depending on nearby items
               
        # Process the ignored items.  this could be done better TODO
        
        for letter in ignorepunct:
            #print(f"searching for char: {letter}")
            for index, srchword in enumerate(mywords):
                if letter in srchword:
                    mywords[index] = srchword.replace(letter,"")
                    #print(f"found letter {letter} in {srchword} - {mywords[index]}")

        ignorewords = ignorecommon.split(",")
        ignoreitems = []
        for index, srchword in enumerate(mywords):
            for ignore in ignorewords:
                #print(f"searching for word: {ignore}")
                if srchword == ignore:
                    #print(f"removing {mywords[index]} from prompt")
                    if index not in ignoreitems:
                        ignoreitems.append(index)
        if len(ignoreitems):
            ignoreitems.sort(reverse=True)
            for reference in ignoreitems:
                #print(f"removed {reference}")
                mywords.pop(reference)
                myweights.pop(reference)

        #now lets strip out double spaces, can't do it earlier
        extraspaces = []
        for index, word in enumerate(mywords):
            if index == 0:
                if word == " ":
                    extraspaces.append(index) 
            else:    
                #print(f"spacecleaning: {index} *{word}*")
                if word == " " and mywords[index-1] == " " and index not in extraspaces:
                    extraspaces.append(index)
                if word == "," and mywords[index-1] == " " and index-1 not in extraspaces:
                    extraspaces.append(index-1)
        if len(extraspaces):
            extraspaces.sort(reverse=True)
            for reference in extraspaces:
                #print(f"removed extraspace {reference}: *{mywords[reference]}*")
                mywords.pop(reference)
                myweights.pop(reference)
        
        #for index, word in enumerate(mywords):
        #    print(f"debug list: *{word}* *{myweights[index]}*")

        prompt = combinewordsandweightsintoprompt(mywords,myweights)
        #print(f"rebuilt prompt: {prompt}")

        full_prompt, tokens = tokenize(prompt)

        for h in range(len(hacks)):
            thishack = hacks[h]
            hackcustom = thishack['custom']
            if hackcustom =="" or hackcustom is None:
                hackcustom = "blank"
            hacks[h]["custom"] = addnotreallytoken(hackcustom)

            if take_x_atonce == 1 and hacks[h] in ["Before and After"]:
                   take_x_atonce = 2 # before and after means 2 tokens are needed.
            #          "Before+After [A:B:V1] (min Group of 2)",
            #         "Fusion Att. Interpolate (A:V1,V2)", "Fusion Linear Interpolate [A,B,C+:V1,V2,V3]",
            #         "Fusion Catmull Interpolate [A,B,C+:V1,V2,V3]", "Fusion Bezier Interpolate [A,B,C+:V1,V2,V3]",
            #          "Fusion [AddA:WhenV1]", "Fusion [RemoveA::WhenV]","Fusion [A:B:StartV1,StopV2]"]

        tokenslength = len(tokens)
        take_x_atonce = min(take_x_atonce, tokenslength)

        #startat and endat are 0-100 %, so we divide by 100, and then round result
        if startat >= endat: # don't do this, if so, ignore it entirely
            startat = 0
            endat = 100
        starttoken = round(tokenslength * startat / 100)  
        # so 0% = length * zero or token 0
        endtoken = round(tokenslength * endat / 100)  
        # so 100% = length * 1 or full tokenlength  (since range is always end-1, this is fine)

        # first generate the potentially 'cleaned' prompt 
        # (which may or may not be different from the original), and label,
        # and setup things to add new images afterward
        clean_prompt,clean_token_ids = tokenize(tokens,True)
        print(f"\n{clean_prompt}")

        if neg_pos == "Positive":
            p.prompt = clean_prompt
        else:
            p.negative_prompt = clean_prompt

        proc = process_images(p)
        if caption_preference != "no caption at all":
            if caption_preference == "visual diff":
                proc.images[0] = promptdiffimg(proc.images[0], prompt, clean_prompt)
            else:            
                proc.images[0] = write_on_image(proc.images[0], "Cleaned Prompt")
        if shared.opts.samples_save:
            images.save_image(
                proc.images[0],
                p.outpath_samples,
                "",
                proc.seed,
                p.prompt,
                shared.opts.samples_format,
                info=proc.infotexts[0],
                p=p)


        # loop to process prompt changes, and generate an image for each prompt.
            
        for g in range(endtoken):  # loops from zero to 1 less than end token, ie every token 0 thru last desired token
            if g < starttoken: # if we are skipping from the start
                continue
            if g > (endtoken - take_x_atonce):  # avoid grabbing last tokens if we're grouping, and not enough are left to grab a full set
                break

            #items to reset each loop
            new_tokens = tokens.copy()

            #process the rest of the prompt, except for the piece under question:
            working_preprompt,returned_pretoken_ids = tokenize(new_tokens[:g],True)
            working_postprompt,returned_posttoken_ids = tokenize(new_tokens[g+take_x_atonce:],True)

            # get the tokens in question:
            working_tokens = new_tokens[g:g+take_x_atonce]
            #print(f"g debug: {working_tokens}")
            if len(working_tokens) == 0:
                continue
            original_prompt, working_tokens = tokenize(working_tokens,True)
            working_prompt = original_prompt
            image_caption = ""

            # loop to process prompt changes, and generate an image for each prompt.
            # so let's iterate thru eaxh of the hacks:
            for h in range(len(hacks)):
                thishack = hacks[h]
                #print(f"{thishack}")
                hackaction = thishack['action']
                if hackaction == "":
                    #nothing selected
                    continue
                hackwhich = thishack['which']
                if hackwhich == "":
                    hackwhich = "Group as One"
                hackwith = thishack['with']
                hackvalue1 = thishack['value1']
                hackvalue2 = thishack['value2']
                hackvalue3 = thishack['value3']
                hackcustomtoken = thishack['custom']   # we've tokenized this earlier

                # Misc variable tweaks
                #
                #"Group as One",
                #"First in Group",
                #"Once in Group",
                #"Randomly in Group",
                #"Last in Group",
                #"Every in Group",
                
                lenwt = len(working_tokens)
                if lenwt == 0:
                    continue
            
                # since python 3.10 is where switch case was added,
                # it's much safer to do the if/elif method
                if hackaction == "Remove":  # Remove is the easy case
                    if hackwhich in ["Group as One", "Every in Group"]:
                        # the nuclear option - remove entire section in question,
                        # this will just wipe it out.
                        working_tokens = []
                        # this really isn't conductive to any
                        # further manipulation so it should be the last function.
                        # Boom.
                    elif hackwhich == "Randomly in Group":
                        sampled = random.sample(working_tokens, k = random.randint(1,lenwt))
                        new_tokens = list(set(working_tokens).difference(sampled))                        
                        working_tokens = new_tokens.copy()
                    elif hackwhich == "Once in Group":
                        sampled = random.choice(working_tokens)
                        working_tokens.remove(sampled)
                    elif hackwhich == "First in Group":
                        working_tokens.pop(0)
                    elif hackwhich == "Last in Group":
                        working_tokens.pop()
                    if lenwt == 0:
                        image_caption = "Removed: " + original_prompt
                        working_prompt = ""
                    else:
                        working_prompt,working_tokens = tokenize(working_tokens,True)
                        #print(f"remove - {working_prompt}")
                elif hackaction == "Shuffle entire group":
                    # we can ignore the which and what, just shuffle the group
                    shufcount = 0
                    original_tokens_order = working_tokens.copy()
                    while shufcount < 5:
                    #    # if 5 mixes fail to find a different order than the input,
                    #    #move on... to handle poor shuffling of small amount of tokens
                         #print(f"shuffling!")
                         random.shuffle(working_tokens)
                         if working_tokens != original_tokens_order:
                            break
                         shufcount += 1
                    working_prompt,working_tokens = tokenize(working_tokens,True)
                    #print(f"shuffle: {working_prompt}")
                    #
                elif hackaction == "Replace":
                    if hackwhich == "Group as One":
                        working_tokens = [pickwith(hackwith,working_tokens[random.randint(1,lenwt)],hackcustomtoken)]
                        # pass list instead of single value, TBD
                    elif hackwhich == "First in Group":
                        working_tokens[0] =  pickwith(hackwith, working_tokens[0], hackcustomtoken)
                    elif hackwhich == "Last in Group":
                        working_tokens[-1] = pickwith(hackwith, working_tokens[-1],hackcustomtoken)
                    elif hackwhich == "Every in Group":
                        for r in range(lenwt):
                            working_tokens[r] = pickwith(hackwith, working_tokens[r],hackcustomtoken)
                    elif hackwhich == "Randomly in Group":
                        sampled = random.sample(working_tokens, k = random.randint(1,lenwt))
                        for r in range(lenwt):
                            if working_tokens[r] in sampled:
                                working_tokens[r] = pickwith(hackwith, working_tokens[r],hackcustomtoken)
                    elif hackwhich == "Once in Group":
                        sampled = random.sample(working_tokens, k = 1)
                        for r in range(lenwt):
                            if working_tokens[r] in sampled:
                                working_tokens[r] = pickwith(hackwith, working_tokens[r],hackcustomtoken)
                    #                       
                    working_prompt,working_tokens = tokenize(working_tokens,True)
                    #
                elif hackaction in ["Add Before", "Add After"]:
                    inserting = working_tokens.copy() 
                    # we have to make a fresh copy to avoid changing working_tokens until the end
                    if hackaction == "Add After":
                        plus = 1 # Add After
                    else:
                        plus = 0 # Add Before
                    if hackwhich == "Group as One":
                        # pass list instead of single value, TBD TODO
                        inserting.insert(0+plus, pickwith(hackwith,working_tokens,hackcustomtoken))
                    elif hackwhich == "Every in Group":
                        for r in range(lenwt):
                            inserting.insert((2*r)+plus, pickwith(hackwith, working_tokens[r],hackcustomtoken))
                    elif hackwhich == "First in Group":
                        inserting.insert( 0+plus, pickwith(hackwith, working_tokens[0] ,hackcustomtoken))
                    elif hackwhich == "Last in Group":
                        inserting.insert(-1+plus, pickwith(hackwith, working_tokens[-1],hackcustomtoken))
                    elif hackwhich == "Once in Group":
                        sampled = random.sample(working_tokens, k = 1)
                        for r in range(lenwt):
                            if working_tokens[r] in sampled:
                                inserting.insert(r+plus, pickwith(hackwith, working_tokens[r],hackcustomtoken))
                    elif hackwhich == "Randomly in Group":
                        sampled = random.sample(working_tokens, k = random.randint(1,lenwt))
                        # we have to work backward on this, since it'll get confusing...
                        for r in reversed(range(len(inserting))):
                            if inserting[r] in sampled:
                                # by checking from the end, if we insert, the early ones are still in right place...
                                inserting.insert(r+plus, pickwith(hackwith, working_tokens[r],hackcustomtoken))
                    #
                    working_prompt, working_tokens = tokenize(inserting,True)
                    #
                    #
                    #elif hackaction == "Before+After [A:B:V1] (min Group of 2)" and len(working_tokens) == 2:

                    # Before and After
                    #    token0,returned_token_id = tokenize([working_tokens[0]],True)
                    #    token1,returned_token_id = tokenize([working_tokens[1]],True)
                    #    working_prompt = "[" + token0.rstrip(" ") + ":" + token1.rstrip(" ") + ":" + str(power) + "] "
                    #    image_caption = working_prompt
                    #    # this really isn't conductive to further manipulation
                    # so it should be one of the last functions

                    #elif hackaction == "Change Weight":
                    #    # TODO do not changing bracketed weights, only parens or carets
                    #    # do we already have a weight already we can change?
                    #   if re.search(r":[0-9\.-]+", working_prompt):
                    #        preexistingweight = True
                    #    else:
                    #        preexistingweight = False
                    #    if hackwhich == "Group as One":
                    #        pass
                    #    elif hackwhich == "Every in Group":
                    #        pass
                    #    elif hackwhich == "Once in Group":
                    #        pass
                    #    elif hackwhich == "First in Group":
                    #        pass
                    #    elif hackwhich == "Last in Group":
                    #        pass
                    #    elif hackwhich == "Randomly in Group":
                    #        pass
                    #       # yes, let's change that one instead of wrapping it with a new one
                    #       working_prompt = re.sub(r":[0-9\.-]+", ':'+str(power), working_prompt)
                    #       # potentially this mean a group won't get a weight if one part is already weighted TODO fix
                    #    else:
                    #       # no weight in current piece so wrap it all with a weight
                    #       working_prompt = f"({working_prompt.rstrip(' ')}:{power}) "
                    #    # this really isn't very conductive to further manipulation
                    # so it should be one of the last functions
                    #    image_caption = working_prompt

                    #"Group as One",
                    #"First in Group",
                    #"Once in Group",
                    #"Randomly in Group",
                    #"Last in Group",
                    #"Every in Group",

                    # TODO         "Before+After [A:B:V1] (min Group of 2)",
                    # "Fusion Att. Interpolate (A:V1,V2)",
                    # "Fusion Linear Interpolate [A,B,C+:V1,V2,V3]",
                    # "Fusion Catmull Interpolate [A,B,C+:V1,V2,V3]",
                    # "Fusion Bezier Interpolate [A,B,C+:V1,V2,V3]",
                    # "Fusion [AddA:WhenV1]",
                    # "Fusion [RemoveA::WhenV]","Fusion [A:B:StartV1,StopV2]"]

                    #if tokenreweight == "Change Weight": # Strengthen or Weaken
                    #    # do we already have a weight already we can change?
                    #    if re.search(r":[0-9\.-]+", working_prompt):
                    # TODO do not changing bracketed weights, only parens or carets
                    #       # yes, let's change that one instead of wrapping it with a new one
                    #       working_prompt = re.sub(r":[0-9\.-]+", ':'+str(power), working_prompt)
                    #       # potentially this mean a group won't get a weight if one part is already weighted TODO fix
                    #    else:
                    #       # no weight in current piece so wrap it all with a weight
                    #       working_prompt = f"({working_prompt.rstrip(' ')}:{power}) "
                    #    # this really isn't very conductive to further manipulation
                    # so it should be one of the last functions
                    #    image_caption = working_prompt

            #put things back together, new_preprompt + working_prompt + new_postprompt
            new_prompt = working_preprompt + " " + working_prompt + " " +working_postprompt
            new_prompt = re.sub(r' +', ' ', new_prompt).strip()

            if image_caption == "":
                if caption_preference == "old part->new part":
                    image_caption = f"{original_prompt} -> {working_prompt}"
                elif caption_preference == "new part only":
                    image_caption = f"{working_prompt}"
            if caption_preference == "entire new prompt":
                image_caption = f"{new_prompt}"
            
            # this puts the prompt up before the generation, useful for console watching
            print(f"\n{new_prompt}")

            if neg_pos == "Positive":
                p.prompt = new_prompt
            else:
                p.negative_prompt = new_prompt

            appendimages = process_images(p)
            proc.images.append(appendimages.images[0])
            proc.infotexts.append(appendimages.infotexts[0])
            if caption_preference != "no caption at all":
                if caption_preference == "visual diff":
                    proc.images[-1] = promptdiffimg(proc.images[-1], clean_prompt, new_prompt)
                else:
                    proc.images[-1] = write_on_image(proc.images[-1], image_caption)
            if shared.opts.samples_save:
                images.save_image(
                proc.images[-1],
                p.outpath_samples,
                "",
                proc.seed,
                p.prompt,
                shared.opts.samples_format,
                info=proc.infotexts[-1],
                p=p)
        #loop end

        # generate the original prompt as given, and label
        # this way the final result is the prompt is recorded
        # as identical to the original given, and restore works correctly.
        # otherwise, the last prompt, no matter how mangled,
        # would be the recorded one, undesired.
        print(f"\n{initial_prompt}")

        if neg_pos == "Positive":
            p.prompt = initial_prompt
        else:
            p.negative_prompt = initial_prompt

        appendimages = process_images(p)
        proc.images.append(appendimages.images[0])
        proc.infotexts.append(appendimages.infotexts[0])
        if caption_preference != "no caption at all":
            proc.images[-1] = write_on_image(proc.images[-1], "Original Prompt")
        if shared.opts.samples_save:
            images.save_image(
                proc.images[-1],
                p.outpath_samples,
                "",
                proc.seed,
                p.prompt,
                shared.opts.samples_format,
                info=proc.infotexts[-1],
                p=p)

        # make grid, if desired, since all images are generated
        grid_flags = self.grid_options_mapping[grid_option]
        unwanted_grid_because_of_img_count = len(proc.images) < 2 and shared.opts.grid_only_if_multiple
        if ((shared.opts.return_grid or shared.opts.grid_save)
            and not p.do_not_save_grid
            and not grid_flags.never_grid
            and not unwanted_grid_because_of_img_count) or grid_flags.always_grid:
            grid = images.image_grid(proc.images)
            proc.images.insert(0,grid)
            proc.infotexts.insert(0, proc.infotexts[0])
            if shared.opts.grid_save or grid_flags.always_save_grid:
                images.save_image(
                    grid,
                    p.outpath_grids,
                    "grid",
                    p.seed,
                    p.prompt,
                    shared.opts.grid_format,
                    info=proc.info,
                    short_filename=not shared.opts.grid_extended_filename,
                    p=p,
                    grid=True)
                    
        return proc
