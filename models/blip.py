import os, json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from models.med import BertConfig, BertModel, BertLMHeadModel
from transformers import BertTokenizer
from models.resnet import blip_resnet

import torch
from torch import nn
import torch.nn.functional as F

from models.transformer import Transformer
from medical_knowledge.GK_knowledge import *
from models.tagencoder import TagEncoder
from models.transformer_poke import Transformer_Poke


CONDITIONS = [
    'enlarged cardiomediastinum',
    'cardiomegaly',
    'lung opacity',
    'lung lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural effusion',
    'pleural other',
    'fracture',
    'support devices',
    'no finding',
]

SCORES = [
'[BLA]',
'[POS]',
'[NEG]',
'[UNC]'
]
node = [
    'normal','other finding','heart','cardiomegaly','spine','scoliosis','pleural','effusion','thickening','pneumothorax',
    'bone', 'bone fractures','lung','emphysema','pneumonia','edema','atelectasis','clcatrix','opacity','lesion',
    'mediastinum','hernia','calcinosis','foreign object','airspace','airspace disease','hypoinflation'
]
nodes = ' '.join(node)
class AddLayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)  
    
    def forward(self, x, sublayer_output):

        residual = x + sublayer_output

        output = self.layer_norm(residual)
        return output

class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 args,
                 tokenizer=None,
                 image_size = 224,
                 prompt = '',
                 ):
        super().__init__()
        self.args = args
        
        vision_width = 2048
        self.visual_encoder = blip_resnet(args)
        
        self.cls_head = nn.Linear(vision_width+512, 18*4)
        # self.cls_head = nn.Linear(512, 18*4)
        nn.init.normal_(self.cls_head.weight, std=0.001)
        if self.cls_head.bias is not None:
            nn.init.constant_(self.cls_head.bias, 0)

        self.vision_proj = nn.Linear(vision_width, 512)
        self.vision_proj_op = nn.Linear(512, vision_width)

        self.tokenizer = tokenizer   
        
        decoder_config = BertConfig.from_json_file('configs/bert_config.json')
        decoder_config.encoder_width = vision_width
        decoder_config.add_cross_attention = True
        decoder_config.is_decoder = True
        self.text_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased',config=decoder_config)

        self.text_decoder.resize_token_embeddings(len(self.tokenizer))
        
        self.prompt = prompt
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1

        self.memory = Transformer(d_model=512,
                                  num_encoder_layers=2,
                                  num_decoder_layers=2,
                                  num_queries=1)
        
        self.explorer = Transformer(d_model=512,
                                  num_encoder_layers=2,
                                  num_decoder_layers=2,
                                  num_queries=1)
        
        self.poke = Transformer_Poke(d_model=512,
                                num_encoder_layers=2,
                                num_decoder_layers=2,
                                num_queries=1)
        
        self.add_ln = AddLayerNorm(d_model=2048)
        self.text_proj = nn.Linear(512, vision_width)
        self.vision_proj2 = nn.Linear(vision_width, 768)
        # if self.args.GK_know:
        c = copy.deepcopy
        attn = MultiHeadedAttention(16, 768)
        ff = PositionwiseFeedForward(768, 1024, 0.1)
        self.cross_attn = Decoder(DecoderLayer(768, c(attn), c(ff), 0.1), 2)

        self.tag_encoder = TagEncoder(0.1, self.args) 
        self.mlpp = nn.Sequential(
                        nn.Linear(768, 1024),  
                        nn.ReLU(),             
                        nn.Linear(1024, 2048)  
                    )
    def forward_for_cam(self, image):
        image_embeds, avg_embeds = self.visual_encoder(image) 
        image_embeds_feature = image_embeds
        avg_embeds = image_embeds
        avg_embeds = avg_embeds[:, -1, :]

        knowledge_gk = nodes
        tag_output = self.tag_encoder(knowledge_gk, image.device)
        image_embeds = self.vision_proj2(image_embeds)
        image_embeds, vis_attn2 = self.cross_attn(image_embeds, tag_output)
        image_embeds = self.mlpp(image_embeds)
        gk_feature = image_embeds
        
        
        return gk_feature

    def forward(self, image, word_embedding, caption, cls_labels, clip_memory, criterion_cls, base_probs):
        image_embeds, avg_embeds = self.visual_encoder(image) 
        image_embeds_feature = image_embeds
        
        knowledge_gk = nodes
        tag_output = self.tag_encoder(knowledge_gk, image.device)
        image_embeds = self.vision_proj2(image_embeds)
        image_embeds, vis_attn2 = self.cross_attn(image_embeds, tag_output)
        image_embeds = self.mlpp(image_embeds)
        gk_feature = image_embeds
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        ##########################
        # NxKxC -> KxNxC
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        query_embed = self.vision_proj(avg_embeds)
        word_embedding = word_embedding.repeat_interleave(query_embed.size(0),dim=0)
        image_poke = self.poke(word_embedding, query_embed, None, query_embed, None) #image_poke=[1,512,8]
        image_poke = image_poke.transpose(1, 2)

        hs = self.memory(clip_memory, None, image_poke, None)

        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        image_poke = image_poke.squeeze(0)
        image_poke = self.vision_proj_op(image_poke)
        avg_embeds = torch.cat((avg_embeds, hs), 1)

        ##########################

        cls_preds = self.cls_head(avg_embeds)
        cls_preds_feature = cls_preds
        cls_preds = cls_preds.view(-1, 4, 18)         
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(image.device)
        loss_cls = criterion_cls(cls_preds, cls_labels)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, return_tensors="pt").to(image.device)
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100) 
        decoder_targets[:,:self.prompt_length] = -100

        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )    ### CausalLMOutputWithCrossAttentions[loss=tensor, logits=tensor[16, 151, 30527]]    
        loss_lm = decoder_output.loss
        return loss_lm, loss_cls, decoder_output.logits, cls_preds_feature, image_embeds_feature, gk_feature

    def interpreter(self, image, caption, cls_labels, cls_preds_int, clip_memory, criterion_cls, base_probs):
        image_embeds, avg_embeds = self.visual_encoder(image) 
        for param in self.memory.parameters():
            param.requires_grad = False
        
        # 冻结视觉投影层
        for param in self.vision_proj.parameters():
            param.requires_grad = False
        
        # 冻结分类头
        for param in self.cls_head.parameters():
            param.requires_grad = False
        query_embed = self.vision_proj(avg_embeds)
        clip_memory = torch.permute(clip_memory, (1, 0, 2))
        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)
        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1) #N*2560
        ##########################

        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18)
        # logit adjustment
        cls_preds[:, 1, :] += torch.log(torch.from_numpy(base_probs)).view(1, -1).to(image.device)
        cls_preds_int = torch.tensor(cls_preds_int, device=image.device, dtype=torch.long)
        loss_cls = criterion_cls(cls_preds, cls_preds_int)
        for param in self.memory.parameters():
            param.requires_grad = True
        
        for param in self.vision_proj.parameters():
            param.requires_grad = True
        
        for param in self.cls_head.parameters():
            param.requires_grad = True
           
        return loss_cls
        
    def generate(self, image, image1, word_embedding, clip_memory, sample=False, num_beams=3, max_length=100, min_length=10, top_p=0.9, repetition_penalty=1.0):
        image_embeds, avg_embeds = self.visual_encoder(image) 

        knowledge_gk = nodes
        tag_output = self.tag_encoder(knowledge_gk, image.device)
        image_embeds = self.vision_proj2(image_embeds)
        image_embeds, vis_attn2 = self.cross_attn(image_embeds, tag_output)
        image_embeds = self.mlpp(image_embeds)

        clip_memory = torch.permute(clip_memory, (1, 0, 2))

        query_embed = self.vision_proj(avg_embeds)

        hs = self.memory(clip_memory, None, query_embed.unsqueeze(0), None)

        # Nx512
        hs = hs.squeeze(0).squeeze(1)
        avg_embeds = torch.cat((avg_embeds, hs), 1)


        # classification branch
        cls_preds = self.cls_head(avg_embeds)
        cls_preds = cls_preds.view(-1, 4, 18)
        cls_preds = F.softmax(cls_preds, dim=1)
        cls_preds_logits = cls_preds[:, 1, :14]
        cls_preds = torch.argmax(cls_preds, dim=1).cpu().numpy().tolist()

        prompts = []
        for j in range(len(cls_preds)):
            prompt = ' '.join([SCORES[c] for c in cls_preds[j]])+' '
            prompts.append(prompt)

        if not sample:
            image_embeds = image_embeds.repeat_interleave(num_beams,dim=0)
        

        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask":image_atts}    

        
        text = self.tokenizer(prompts, return_tensors="pt")
        input_ids = text.input_ids.to(image.device)
        attn_masks = text.attention_mask.to(image.device)
        input_ids[:,0] = self.tokenizer.bos_token_id
        input_ids = input_ids[:, :-1] 
        attn_masks = attn_masks[:, :-1] 

        #beam search
        outputs = self.text_decoder.generate(input_ids=input_ids,
                                             min_length=min_length, # 4.25 Transformers
                                             max_new_tokens=max_length,
                                             num_beams=num_beams,
                                             eos_token_id=self.tokenizer.sep_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id, 
                                             repetition_penalty=repetition_penalty,
                                             attention_mask = attn_masks,
                                             **model_kwargs)       
             
        # print(outputs)
        # print(outputs.size()) #[N, 169]
        # exit()
        captions = []    
        for i, output in enumerate(outputs):
            caption = self.tokenizer.decode(output, skip_special_tokens=True)    
            # print(caption)
            captions.append(caption[len(prompts[i]):])
        # print(captions)
        # exit()
        return captions, cls_preds, cls_preds_logits

def blip_decoder(args, tokenizer, **kwargs):

    model = BLIP_Decoder(args, tokenizer, **kwargs)
    return model    
    
