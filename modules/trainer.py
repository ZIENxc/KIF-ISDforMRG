import os
from abc import abstractmethod

import time
import torch
import torch.distributed as dist
import pandas as pd
import numpy as np
from numpy import inf
from .metrics_clinical import CheXbertMetrics
import copy
from .optims import LinearWarmupCosineLRScheduler
import clip
from transformers import BertTokenizer
from torch import nn
import torch.nn.functional as F


class BaseTrainer(object):
    def __init__(self, model, tmodel, criterionKD, criterion_cls, base_probs, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model = model
        self.tmodel = tmodel
        self.device = device
        self.is_main_process = is_main_process

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)
        self.criterion_cls = criterion_cls
        self.criterionKD = criterionKD ##
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns
        #################
        self.optimizer = None
        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue  # frozen weights
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print("number of trainable parameters: {}".format(num_parameters))
        optim_params = [
            {
                "params": p_wd,
                "weight_decay": float(self.args.weight_decay),
            },
            {"params": p_non_wd, "weight_decay": 0},
        ]
        beta2 = 0.999
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, beta2),
        )
        #################

        self.epochs = self.args.epochs

        self.mnt_metric = 'test_' + args.monitor_metric

        self.mnt_best = 0 
        self.log_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                # for different shuffling
                self.train_dataloader.sampler.set_epoch(epoch)

            result = self._train_epoch_blip(epoch)
            # dist.barrier()
            result = self.eval_blip(result)

            # save logged information 
            log = {'epoch': epoch}
            log.update(result)

            # record best
            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    self.mnt_best = log[self.mnt_metric]
                    self.log_best = copy.deepcopy(log)
                    best_path = os.path.join(self.checkpoint_dir, 'model_best_t_gk4.26.pth')
                    torch.save(self.model.state_dict(), best_path)
                    print("Saving current best to {}".format(best_path))

            # print logged information 
            for key, value in log.items():
                print('\t{:15s}: {}'.format(str(key), value))

        if self.is_main_process:
            print('Best results w.r.t {}:'.format(self.mnt_metric))
            for key, value in self.log_best.items():
                print('\t{:15s}: {}'.format(str(key), value))

def createCLIP():
    pretrained_path = './modules/clip-imp-pretrained_128_6_after_4.pt'
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    # Load pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    state_dict = torch.load(pretrained_path, map_location="cuda:0")
    model.load_state_dict(state_dict)
    print("load checkpoint from {}".format(pretrained_path))

    return model

def process_long_texts(reports, clip_model, device, max_length=77):
    batch_size = len(reports)
    all_text_features = []

    for report in reports:

        segments = []
        words = report.split()  
        current_segment = []
        current_length = 0

        for word in words:

            if current_length + len(word) + 1 > max_length:  
                segments.append(" ".join(current_segment))  
                current_segment = [word]  
                current_length = len(word)
            else:
                current_segment.append(word)
                current_length += len(word) + 1  

        
        if current_segment:
            segments.append(" ".join(current_segment))

        
        segment_features = []
        for segment in segments:
            
            text_tokens = clip.tokenize(segment, truncate=True).to(device)
            with torch.no_grad():
                text_feature = clip_model.encode_text(text_tokens)
                text_feature = text_feature.float()
                text_feature /= text_feature.norm(dim=-1, keepdim=True)
            segment_features.append(text_feature)

        
        if segment_features:
            combined_feature = torch.mean(torch.stack(segment_features), dim=0)  # [feature_dim]
        else:
            combined_feature = torch.zeros(clip_model.text_projection.shape[1]).to(device)  

        all_text_features.append(combined_feature)

 
    all_text_features = torch.stack(all_text_features)  # [batch_size, feature_dim]
    return all_text_features

class Trainer(BaseTrainer):
    def __init__(self, model, tmodel, criterionKD, criterion_cls, base_probs, metric_ftns, args, train_dataloader, val_dataloader, test_dataloader, device, is_main_process):
        super(Trainer, self).__init__(model, tmodel, criterionKD, criterion_cls, base_probs, metric_ftns, args, device, is_main_process)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer, 
            self.args.epochs, 
            self.args.min_lr, 
            self.args.init_lr, 
            decay_rate=None, 
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )
        self.clip = createCLIP()
        self.tokenizer = BertTokenizer.from_pretrained(
                        "bert-base-uncased",
                        additional_special_tokens=["[BLA]", "[POS]", "[NEG]", "[UNC]"]
)
        

    def _train_epoch_blip(self, epoch):
        train_loss = 0
        self.model.train()
        self.tmodel.eval()

        with torch.no_grad():   
            reports = 'normal, cardiomegaly, scoliosis, fractures, effusion, thickening, pneumothorax, hernia, calcinosis, emphysema, pneumonia, edema, atelectasis, cicatrix, opacity, lesion, airspace disease, hypoinflation, medical device, other'
            texts = clip.tokenize(reports, truncate=True).to(self.device)  
            text_features = self.clip.encode_text(texts)
            text_features = text_features.float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            word_embedding = text_features
            
        loss_inter = 0
        loss_mid = 0
        for batch_idx, (images, images_1, captions, cls_labels, clip_memory) in enumerate(self.train_dataloader):
            images = images.to(self.device) #[16,3,224,224]
            images_1 = images_1.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = clip_memory.to(self.device)
            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx)
            loss_lm, loss_cls, logits, cls_feature, img_feature, gk_feature = self.model(images, word_embedding, captions, cls_labels, clip_memory, self.criterion_cls, self.base_probs)
            _, _, t_logits, cls_feature_t, img_feature_t, gk_feature_t = self.tmodel(images, word_embedding, captions, cls_labels, clip_memory, self.criterion_cls, self.base_probs)
            kd_loss = self.criterionKD(logits, t_logits.detach()) * 0.01
            
            img_feature_t_flat = img_feature_t.contiguous().view(16, 49 * 2048)
            img_feature_flat = img_feature.contiguous().view(16, 49 * 2048)
            img_feature_t_flat_T = img_feature_t_flat.T
            img_feature_flat_T = img_feature_flat.T
            cls_feature_t_T = cls_feature_t.T
            cls_feature_T = cls_feature.T
            Hf_t = torch.matmul(img_feature_t_flat, img_feature_t_flat_T)  # [16, 16]
            Hf = torch.matmul(img_feature_flat, img_feature_flat_T)
            Hc = torch.matmul(cls_feature, cls_feature_T)                  # [16, 16]
            Hc_t = torch.matmul(cls_feature_t, cls_feature_t_T)
            H = torch.cat((Hf, Hc), dim=1)
            H_t = torch.cat((Hf_t, Hc_t), dim=1)
            H_norm = H / torch.norm(H, p=2, dim=1, keepdim=True)
            H_norm_t = H_t / torch.norm(H_t, p=2, dim=1, keepdim=True)
            loss_mid = torch.norm((H_norm - H_norm_t), p=2, dim=1, keepdim=False)
            loss_mid = torch.sum(loss_mid, dim=0)


            #####Interpreter
            if batch_idx % 100 == 1000000:  
                self.model.eval()
                reports, cls_preds_int, _ = self.model.generate(images, images_1, word_embedding, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)

                clip_inter = process_long_texts(reports, self.clip, self.device)
                loss_inter = self.model.interpreter(images, captions, cls_labels, cls_preds_int, clip_inter, self.criterion_cls, self.base_probs)
                loss = loss_lm + self.args.cls_weight * loss_cls + kd_loss + loss_inter
                self.model.train()
            #####
            # exit()
            else: 
                loss = loss_lm + self.args.cls_weight*loss_cls + kd_loss + loss_mid
                # loss = loss_lm + self.args.cls_weight*loss_cls
            if batch_idx%1000 == 0:
                print("{}/{} loss: {}, loss_lm: {}, loss_cls: {}, kd_loss: {}, loss_mid: {}".format(batch_idx, len(self.train_dataloader), loss.item(), loss_lm.item(), self.args.cls_weight*loss_cls.item(), kd_loss.item(), loss_mid))
            train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()
            # break
        log = {'train_loss': train_loss / len(self.train_dataloader)}

        return log

    def eval_blip(self, log):
        self.model.eval()

        logits = []
        counts = []
        with torch.no_grad():
            reports = 'normal, cardiomegaly, scoliosis, fractures, effusion, thickening, pneumothorax, hernia, calcinosis, emphysema, pneumonia, edema, atelectasis, cicatrix, opacity, lesion, airspace disease, hypoinflation, medical device, other'
            texts = clip.tokenize(reports, truncate=True).to(self.device)  
            text_features = self.clip.encode_text(texts)
            text_features = text_features.float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            word_embedding = text_features
        with torch.no_grad():
            val_gts, val_res = [], []
            for batch_idx, (images, images_1, captions, cls_labels, clip_memory) in enumerate(self.val_dataloader):
                images = images.to(self.device) 
                images_1 = images_1.to(self.device)
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)
                ground_truths = captions
                reports, cls_preds, cls_preds_logits = self.model.generate(images, images_1, word_embedding, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
                ## logit adjustment
                cls_labels = (cls_labels==1).float()
                logit = cls_preds_logits*cls_labels
                logits.append(logit.cpu().numpy())
                counts.append(cls_labels.cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(ground_truths)

            #######
            logits = np.concatenate(logits, axis=0)
            counts = np.concatenate(counts, axis=0)
            logits = np.sum(logits, 0)
            counts = np.sum(counts, 0)
            logits = logits / counts
            logits /= np.max(logits)
            logits = np.append(logits, [1, 1, 1, 1]) # 4 auxiliary diseases
            #######
            self.base_probs = logits # update class distribution
            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts)},
                                       {i: [re] for i, re in enumerate(val_res)})
            val_ce = self.chexbert_metrics.compute(val_gts, val_res)
            log.update(**{'val_' + k: v for k, v in val_met.items()})
            log.update(**{'val_' + k: v for k, v in val_ce.items()})

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, images_1, captions, cls_labels, clip_memory) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                images_1 = images_1.to(self.device)
                cls_labels = cls_labels.numpy().tolist()
                clip_memory = clip_memory.to(self.device) 
                ground_truths = captions
                reports, _, _ = self.model.generate(images, images_1, word_embedding, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len) ##output=[16, 161], token_id

                test_res.extend(reports)
                test_gts.extend(ground_truths)
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
        return log

    
