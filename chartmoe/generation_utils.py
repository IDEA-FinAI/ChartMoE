"""
    FEATURE: Generation Scipt of ChartMoE
    AUTHOR: Brian Qu
    URL: https://arxiv.org/abs/2409.03277
"""
import os

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision

from chartmoe.utils.custom_path import ChartMoE_HF_PATH

def __padding__(image):
    width, height = image.size
    tar = max(width, height)
    top_padding = int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = int((tar - width)/2)
    right_padding = tar - width - left_padding
    image = torchvision.transforms.functional.pad(image, [left_padding, top_padding, right_padding, bottom_padding])
    return image
    
class ChartMoE_Robot:
    def __init__(self, ckpt_path = None, img_padding = False):
        model_path = ckpt_path if ckpt_path else ChartMoE_HF_PATH
        tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                trust_remote_code=True
            )
        print(f"\033[34mLoad model from {model_path}\033[0m")
        self.model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                ).half().cuda().eval()
        self.tokenizer = tokenizer
        self.model.tokenizer = tokenizer
        
        self.prompt = '[UNUSED_TOKEN_146]user\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

        self.img_padding = img_padding
    
    def reset_prompt(self, prompt):
        self.prompt = prompt
    
    def chat(
            self, 
            image_path=None,
            image=None, 
            question="", 
            history="",
            temperature=1, 
            max_new_tokens=1000, 
            num_beams=1,
            do_sample=False, 
            repetition_penalty=1.0,
        ):
        need_bos = True
        pt1 = 0
        embeds = []
        im_mask = []
        question = self.prompt.format(question)
        history += question

        if image_path and image:
            assert False, "Just give the `image_path` or give the `PIL.Image` to `image`!"
        if image_path is None and image is None:
            assert False, "`image_path` and `image` are both None! Please give the `image_path` or give the `PIL.Image` to `image`!"
        
        if image_path:
            images = [image_path]
        else:
            images = [image]
        images_loc = [0]

        for i, pts in enumerate(images_loc + [len(history)]):
            subtext = history[pt1:pts]
            if need_bos or len(subtext) > 0:
                text_embeds = self.model.encode_text(subtext, add_special_tokens=need_bos)
                embeds.append(text_embeds)
                im_mask.append(torch.zeros(text_embeds.shape[:2]).cuda())
                need_bos = False
            if i < len(images):
                try:
                    image = Image.open(images[i]).convert('RGB')
                except:
                    image = images[i].convert('RGB')
                if self.img_padding:
                    image = __padding__(image)
                image = self.model.vis_processor(image).unsqueeze(0).cuda()
                image_embeds = self.model.encode_img(image)
                embeds.append(image_embeds)
                im_mask.append(torch.ones(image_embeds.shape[:2]).cuda())
            pt1 = pts
        embeds = torch.cat(embeds, dim=1)
        im_mask = torch.cat(im_mask, dim=1)
        im_mask = im_mask.bool()

        eos_token_id = [
            self.tokenizer.convert_tokens_to_ids(['[UNUSED_TOKEN_145]'])[0],
            self.tokenizer.eos_token_id,
        ]
        outputs = self.model.generate(
                    inputs_embeds=embeds,
                    im_mask=im_mask,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample, 
                    repetition_penalty=repetition_penalty,
                    eos_token_id=eos_token_id,
                )

        output_token = outputs[0]
        if output_token[0] == 0 or output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.tokenizer.decode(output_token, add_special_tokens=False)
        history += output_text
        output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
        return output_text, history