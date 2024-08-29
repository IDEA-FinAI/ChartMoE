import os

import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import torchvision

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
    def __init__(self, img_padding = False):
        tokenizer = AutoTokenizer.from_pretrained(
                '/data/FinAi_Mapping_Knowledge/qiyiyan/qbw/cache/ckpt/chartmoe/chartmoe_hf', 
                trust_remote_code=True
            )
        self.model = AutoModel.from_pretrained(
                    '/data/FinAi_Mapping_Knowledge/qiyiyan/qbw/cache/ckpt/chartmoe/chartmoe_hf',
                    trust_remote_code=True
                ).cuda().half().eval()
        self.tokenizer = tokenizer
        self.model.tokenizer = tokenizer
        
        self.prompt = '[UNUSED_TOKEN_146]user\n{}[UNUSED_TOKEN_145]\n[UNUSED_TOKEN_146]assistant\n'

        self.img_padding = img_padding
    
    def reset_prompt(self, prompt):
        self.prompt = prompt
    
    def chat(
            self, 
            image_path, 
            question, 
            history="",
            temperature=1, 
            max_new_tokens=1000, 
            num_beams=3,
            do_sample=False, 
            repetition_penalty=1.0,
        ):
        need_bos = True
        pt1 = 0
        embeds = []
        im_mask = []
        images = [image_path]
        images_loc = [0]
        question = self.prompt.format(question)
        history += question

        for i, pts in enumerate(images_loc + [len(question)]):
            subtext = question[pt1:pts]
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

        outputs = self.model.generate(
                    inputs_embeds=embeds,
                    im_mask=im_mask,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    num_beams=num_beams,
                    do_sample=do_sample, 
                    repetition_penalty=repetition_penalty
                )

        output_token = outputs[0]
        if output_token[0] == 0 or output_token[0] == 1:
            output_token = output_token[1:]
        output_text = self.model.tokenizer.decode(output_token, add_special_tokens=False)
        history += output_text
        output_text = output_text.split('[UNUSED_TOKEN_145]')[0].strip()
        return output_text, history