import logging
import os
from abc import abstractmethod
import numpy as np
import time

import cv2
import torch
import clip
from .metrics_clinical import CheXbertMetrics

class BaseTester(object):
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics('./checkpoints/stanford/chexbert/chexbert.pth', args.batch_size, device)

        logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                            datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
        self.logger = logging.getLogger(__name__)


        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns

        self.epochs = self.args.epochs
        self.save_dir = self.args.save_dir

    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def plot(self):
        raise NotImplementedError

def createCLIP():
    pretrained_path = './modules/clip-imp-pretrained_128_6_after_4.pt'
    device = "cuda:0" if torch.cuda.is_available() else "cpu" 
    # Load pre-trained CLIP model
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    state_dict = torch.load(pretrained_path, map_location="cuda:0")
    model.load_state_dict(state_dict)
    print("load checkpoint from {}".format(pretrained_path))

    return model

class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super(Tester, self).__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader
        self.clip = createCLIP()

    def test_blip(self):
        self.logger.info('Start to evaluate in the test set.')
        log = dict()
        self.model.eval()
        with torch.no_grad():   
            reports = 'normal, cardiomegaly, scoliosis, fractures, effusion, thickening, pneumothorax, hernia, calcinosis, emphysema, pneumonia, edema, atelectasis, cicatrix, opacity, lesion, airspace disease, hypoinflation, medical device, other'
            texts = clip.tokenize(reports, truncate=True).to(self.device)  
            text_features = self.clip.encode_text(texts)
            text_features = text_features.float()
            text_features /= text_features.norm(dim=-1, keepdim=True)
            word_embedding = text_features
        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, (images, images_1, captions, cls_labels, clip_memory, image_o) in enumerate(self.test_dataloader):
                images = images.to(self.device) 
                images_1 = images_1.to(self.device)
                clip_memory = clip_memory.to(self.device) 
                ground_truths = captions
                # print(1)
                reports, _, _ = self.model.generate(images, images_1, word_embedding, clip_memory, sample=False, num_beams=self.args.beam_size, max_length=self.args.gen_max_len, min_length=self.args.gen_min_len)
                if(batch_idx < 10):
                    print(image_o)
                    print("reports:")
                    print(reports)
                    print("ground_truths:")
                    print(ground_truths)
                    print(cls_labels)
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                if batch_idx % 10 == 0:
                    print('{}/{}'.format(batch_idx, len(self.test_dataloader)))
            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            log.update(**{'test_' + k: v for k, v in test_ce.items()})
        return log

