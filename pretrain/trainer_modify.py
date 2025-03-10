import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dataloader import PretrainEmoDataset, SecondPhaseEmoDataset, RandomBucketSampler, StandardSampler, BaselineDataset
from torch.utils import data
from modules.FeatureFuser import Wav2vecWrapper, Wav2vec2Wrapper, Wav2vec2PretrainWrapper
from modules.NN import LinearHead, RNNLayer
from utils.metrics import ConfusionMetrics
import pytorch_lightning.core.lightning as pl



class PretrainedEmoClassifier(pl.LightningModule):
    def __init__(self, max_epochs, batch_size, lr, datadir, labeldir, modelpath, labeling_method, valid_split):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wav2vec = Wav2vecWrapper(modelpath)
        self.data = PretrainEmoDataset(datadir, labeldir, labeling_method=labeling_method)
        self.linearhead = LinearHead(1024, 128, self.data.nemos)
        self.linearhead_rnn = LinearHead(1024, 128, self.data.nemos)
        self.rnn = RNNLayer(1024, 1024)
        numtraining = int(len(self.data) * valid_split)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))
        counter = self.data.emos
        weights = torch.tensor(
            [counter[c] for c in self.data.emoset]
        ).float()
        weights = weights.sum() / weights
        weights = weights / weights.sum()
        print(
            f"Weigh losses by prior distribution of each class: {weights}."
        )
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        reps = self.wav2vec(x)
        logits_reduced = self.linearhead_rnn(self.rnn(reps))
        logits = self.linearhead(reps)
        return logits, logits_reduced

    def train_dataloader(self):
        return data.DataLoader(self.traindata,
                               batch_size=self.batch_size,
                               num_workers=8,
                               drop_last=True,
                               shuffle=True,
                               collate_fn=self.data.truncCollate)

    def val_dataloader(self):
        return data.DataLoader(self.valdata,
                               batch_size=self.batch_size,
                               num_workers=8,
                               drop_last=False,
                               shuffle=False,
                               collate_fn=self.data.truncCollate)

    def configure_optimizers(self):
        parameters = list(self.linearhead.parameters()) + list(self.wav2vec.trainable_params()) + \
                     list(self.rnn.parameters()) + list(self.linearhead_rnn.parameters())
        optimizer = optim.AdamW(parameters, lr=self.lr)
        #Learning rate scheduler
        num_training_steps = self.max_epochs
        num_warmup_epochss = int(0.05 * num_training_steps)
        num_flat_steps = int(0.05 * num_training_steps)
        def lambda_lr(current_step: int):
            if current_step < num_warmup_epochss:
                return float(current_step) / float(max(1, num_warmup_epochss))
            elif current_step < (num_warmup_epochss + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_epochss + num_flat_steps)))
            )
        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lambda_lr),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        feats, label = batch
        output = self(feats)
        if output is None:
            raise ValueError("forward() returned None.")
        try:
            pout, pout_reduced = output
        except Exception as e:
            raise ValueError(f"Error unpacking forward() output: {e}")

        if pout_reduced is None:
            raise ValueError("pout_reduced is None.")

        loss1 = self.criterion(pout_reduced, label)
        if loss1 is None:
            raise ValueError("Loss from criterion (pout_reduced vs. label) is None.")
        
        label_expanded = label.unsqueeze(1).expand(-1, pout.size(1)).reshape(-1)
        pout_reshaped = pout.reshape(-1, pout.size(2))
        loss2 = self.criterion(pout_reshaped, label_expanded)
        if loss2 is None:
            raise ValueError("Loss from criterion (pout vs. expanded label) is None.")

        loss = loss1 + loss2
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        # 나머지 코드...
        return loss

    def validation_step(self, batch, batch_idx):
        feats, label = batch
        pout, pout_reduced = self(feats)
        loss = self.criterion(pout_reduced, label)
        label = label.unsqueeze(1).expand(-1, pout.size(1)).reshape(-1)
        pout = pout.reshape(-1, pout.size(2))
        loss += self.criterion(pout, label)
        acc = (label == torch.argmax(pout, -1)).float().mean()
        validdict = {
            'valid_loss': loss,
            'valid_acc': acc
        }
        self.log_dict(validdict, on_epoch=True, logger=True)
        return loss

class MinimalClassifier(pl.LightningModule):
    def __init__(self, backend='wav2vec2', wav2vecpath=None):
        assert backend in ['wav2vec2', 'wav2vec']
        super().__init__()
        self.backend = backend
        if backend == 'wav2vec2':
            self.wav2vec2 = Wav2vec2Wrapper(pretrain=False)
        else:
            assert wav2vecpath is not None
            self.wav2vec = Wav2vecWrapper(wav2vecpath)

    def forward(self, x, length=None):
        reps = getattr(self, self.backend)(x, length)
        return reps

class SecondPassEmoClassifier(pl.LightningModule):
    def __init__(self, max_epochs, batch_size,
                 lr, warmup_epochs, nclusters, maxseqlen,
                 datadir, unsupdatadir, labeldir, distributed,
                 use_bucket_sampler, train_bucket_size, val_bucket_size, dynamic_batch, valid_split):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.dynamic_batch = dynamic_batch
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.distributed = distributed
        self.use_bucket_sampler = use_bucket_sampler
        self.train_bucket_size = train_bucket_size
        self.val_bucket_size = val_bucket_size
        self.wav2vec2 = Wav2vec2Wrapper(pretrain=True)
        self.maxseqlen = maxseqlen
        self.linearheads = nn.ModuleList([LinearHead(1024, 256, ncluster) for i, ncluster in enumerate(nclusters)])
        self.data = SecondPhaseEmoDataset(datadir, unsupdatadir, labeldir, maxseqlen=maxseqlen,
                                          final_length_fn=self.wav2vec2.get_feat_extract_output_lengths)
        numtraining = int(len(self.data) * valid_split)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x, length):
        reps, maskidx = self.wav2vec2(x, length)
        to_train = reps[maskidx] #L', C
        logits = []
        for lh in self.linearheads:
            logits.append(lh(to_train))
        return logits, maskidx

    def train_dataloader(self):
        idxs = self.traindata.indices
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [l if l < self.maxseqlen else self.maxseqlen for l in length]
        sampler = RandomBucketSampler(self.train_bucket_size, length, self.batch_size, drop_last=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank, dynamic_batch=self.dynamic_batch) if self.use_bucket_sampler else \
                  data.BatchSampler(
                      StandardSampler(self.traindata, shuffle=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank, dynamic_batch=self.dynamic_batch),
                      self.batch_size, drop_last=True
                  )
        return data.DataLoader(self.traindata,
                               num_workers=8,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def val_dataloader(self):
        
        idxs = self.valdata.indices
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [l if l < self.maxseqlen else self.maxseqlen for l in length]
        sampler = RandomBucketSampler(self.val_bucket_size, length, self.batch_size, drop_last=False, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank, dynamic_batch=self.dynamic_batch) if self.use_bucket_sampler else \
                  data.BatchSampler(
                      StandardSampler(self.valdata, shuffle=False, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank, dynamic_batch=self.dynamic_batch),
                      self.batch_size, drop_last=False
                  )
        return data.DataLoader(self.valdata,
                               num_workers=8,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def configure_optimizers(self):
        optimizer = optim.AdamW(list(self.linearheads.parameters()) + list(self.wav2vec2.trainable_params()), lr=self.lr)
        #Learning rate scheduler
        num_training_steps = self.max_epochs
        num_warmup_epochs = self.warmup_epochs
        num_flat_steps = int(0.05 * num_training_steps)
        def lambda_lr(current_step: int):
            if current_step < num_warmup_epochs:
                return float(current_step) / float(max(1, num_warmup_epochs))
            elif current_step < (num_warmup_epochs + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_epochs + num_flat_steps)))
            )
        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lambda_lr),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # 배치를 언팩합니다.
        try:
            feats, labels, length = batch
        except Exception as e:
            raise ValueError(f"Batch unpacking error: {e}. Expected three elements (feats, labels, length) but got {len(batch)}.")

        # 모델 forward를 호출합니다.
        output = self(feats, length=length)
        if output is None:
            raise ValueError("Forward() returned None.")

        # forward()는 (pouts, maskidx)를 반환해야 합니다.
        try:
            pouts, maskidx = output
        except Exception as e:
            raise ValueError(f"Error unpacking forward() output: {e}")

        if not pouts:
            raise ValueError("Forward() returned an empty list for pouts.")

        # 초기 loss와 acc는 같은 device의 0 텐서로 초기화합니다.
        loss = torch.tensor(0.0, device=feats.device)
        acc = torch.tensor(0.0, device=feats.device)

        # 각 출력에 대해 손실과 정확도를 계산합니다.
        for i, logit in enumerate(pouts):
            try:
                current_label = labels[:, i][maskidx]
            except Exception as e:
                raise ValueError(f"Error extracting label for output index {i}: {e}")
            
            current_loss = self.criterion(logit, current_label)
            if current_loss is None:
                raise ValueError(f"Criterion returned None for output index {i}.")
            loss = loss + current_loss

            # 예측 및 정확도 계산
            pred = torch.argmax(logit, dim=-1)
            current_acc = (current_label == pred).float().mean()
            acc = acc + current_acc

        # pouts 개수만큼 정확도 평균 계산
        acc = acc / len(pouts)

        # 로그를 기록합니다.
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss


    def validation_step(self, batch, batch_idx):
        feats, labels, length = batch
        pouts, maskidx = self(feats, length=length)
        loss, acc = 0, 0
        for i, logit in enumerate(pouts):
            label = labels[:, i][maskidx]
            loss += self.criterion(logit, label)
            acc += (label == torch.argmax(logit, -1)).float().mean()
        acc /= len(pouts)
        validdict = {
            'valid_loss': loss,
            'valid_acc': acc
        }
        self.log_dict(validdict, on_epoch=True, logger=True, sync_dist=self.distributed)
        return loss

class PretrainedRNNHead(pl.LightningModule):
    def __init__(self, n_classes, backend='wav2vec2', wav2vecpath=None):
        assert backend in ['wav2vec2', 'wav2vec']
        super().__init__()
        self.backend = backend
        if backend == 'wav2vec2':
            self.wav2vec2 = Wav2vec2Wrapper(pretrain=False)
            feature_dim = 1024
        else:
            assert wav2vecpath is not None
            self.wav2vec = Wav2vecWrapper(wav2vecpath)
            feature_dim = 512
        self.rnn_head = nn.LSTM(feature_dim, 256, 1, bidirectional=True)
        self.linear_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1024, n_classes)
        )

    def trainable_params(self):
        return list(self.rnn_head.parameters()) + list(self.linear_head.parameters()) + list(getattr(self, self.backend).trainable_params())

    def forward(self, x, length):
        reps = getattr(self, self.backend)(x, length)
        last_feat_pos = getattr(self, self.backend).get_feat_extract_output_lengths(length) - 1
        logits = reps.permute(1, 0, 2) #L, B, C
        masks = torch.arange(logits.size(0), device=logits.device).expand(last_feat_pos.size(0), -1) < last_feat_pos.unsqueeze(1)
        masks = masks.float()
        logits = (logits * masks.T.unsqueeze(-1)).sum(0) / last_feat_pos.unsqueeze(1)
        logits = self.linear_head(logits)
        return logits

class ContinueFinetuningBaseline(pl.LightningModule):
    def __init__(self, max_epochs, batch_size,
                 lr, warmup_epochs, maxseqlen, nclusters,
                 datadir, labelpath, distributed,
                 use_bucket_sampler, train_bucket_size, val_bucket_size,
                 use_additional_obj):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.warmup_epochs = warmup_epochs
        self.distributed = distributed
        self.use_bucket_sampler = use_bucket_sampler
        self.train_bucket_size = train_bucket_size
        self.val_bucket_size = val_bucket_size
        self.wav2vec2 = Wav2vec2PretrainWrapper()
        self.use_additional_obj = use_additional_obj
        self.maxseqlen = maxseqlen
        if use_additional_obj:
            self.linearheads = nn.ModuleList([nn.Linear(self.wav2vec2.wav2vec2PT.config.proj_codevector_dim, ncluster) for i, ncluster in enumerate(nclusters)])
            self.data = SecondPhaseEmoDataset(datadir, None, labelpath, maxseqlen=maxseqlen,
                                              final_length_fn=self.wav2vec2.get_feat_extract_output_lengths)
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.data = BaselineDataset(datadir, labelpath, maxseqlen=maxseqlen)
        numtraining = int(len(self.data) * 1.0)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))

    def train_dataloader(self):
        idxs = self.traindata.indices
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [l if l < self.maxseqlen else self.maxseqlen for l in length]
        sampler = RandomBucketSampler(self.train_bucket_size, length, self.batch_size, drop_last=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank) if self.use_bucket_sampler else \
                  data.BatchSampler(
                      StandardSampler(self.traindata, shuffle=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank),
                      self.batch_size, drop_last=True
                  )
        return data.DataLoader(self.traindata,
                               num_workers=8,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def val_dataloader(self):
        
        idxs = self.valdata.indices
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [l if l < self.maxseqlen else self.maxseqlen for l in length]
        sampler = RandomBucketSampler(self.val_bucket_size, length, self.batch_size, drop_last=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank) if self.use_bucket_sampler else \
                  data.BatchSampler(
                      StandardSampler(self.valdata, shuffle=False, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank),
                      self.batch_size, drop_last=False
                  )
        return data.DataLoader(self.valdata,
                               num_workers=8,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def configure_optimizers(self):
        parameters = self.wav2vec2.trainable_params()
        if self.use_additional_obj:
            parameters = list(parameters) + list(self.linearheads.parameters())
        optimizer = optim.AdamW(parameters, lr=self.lr)
        #Learning rate scheduler
        num_training_steps = self.max_epochs
        num_warmup_epochss = self.warmup_epochs
        num_flat_steps = int(0.05 * num_training_steps)
        def lambda_lr(current_step: int):
            if current_step < num_warmup_epochss:
                return float(current_step) / float(max(1, num_warmup_epochss))
            elif current_step < (num_warmup_epochss + num_flat_steps):
                return 1.0
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - (num_warmup_epochss + num_flat_steps)))
            )
        scheduler = {
            'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lambda_lr),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def forward(self, x, length):
        reps = self.wav2vec2(x, length)
        # reps는 객체로서, 여기서 projected_states와 loss 속성이 있어야 함
        try:
            reps, pretrain_loss = reps.projected_states, reps.loss
        except AttributeError as e:
            raise ValueError(f"Forward output structure error: {e}")

        if not self.use_additional_obj:
            # 학습 모드에서 loss 값이 반드시 반환되어야 함
            if pretrain_loss is None:
                raise ValueError("pretrain_loss is None.")
            return pretrain_loss
        else:
            to_train = reps.reshape(-1, reps.size(2))
            logits = []
            for lh in self.linearheads:
                logits.append(lh(to_train))
            if pretrain_loss is None or logits is None:
                raise ValueError("Either pretrain_loss or logits is None.")
        return logits, pretrain_loss


    def training_step(self, batch, batch_idx):
        if self.use_additional_obj:
            # 추가 객체(use_additional_obj=True)일 때 배치 언팩
            try:
                feats, labels, length = batch
            except Exception as e:
                raise ValueError(f"[use_additional_obj] Batch unpacking error: {e}. Expected (feats, labels, length).")
            
            # 모델 forward 호출
            output = self(feats, length=length)
            if output is None:
                raise ValueError("[use_additional_obj] forward() returned None.")
            try:
                pouts, loss = output
            except Exception as e:
                raise ValueError(f"[use_additional_obj] Error unpacking forward() output: {e}")
            if not pouts:
                raise ValueError("[use_additional_obj] forward() returned an empty list for outputs (pouts).")
            
            # 각 출력에 대해 손실을 추가로 계산
            for i, logit in enumerate(pouts):
                try:
                    current_label = labels[:, i].reshape(-1)
                except Exception as e:
                    raise ValueError(f"[use_additional_obj] Error processing label at index {i}: {e}")
                current_loss = self.criterion(logit, current_label)
                if current_loss is None:
                    raise ValueError(f"[use_additional_obj] Criterion returned None for output index {i}.")
                loss = loss + current_loss

        else:
            # use_additional_obj가 False인 경우 배치 언팩
            try:
                feats, length = batch
            except Exception as e:
                raise ValueError(f"[non-additional] Batch unpacking error: {e}. Expected (feats, length).")
            loss = self(feats, length=length)
            if loss is None:
                raise ValueError("[non-additional] forward() returned None.")
        
        # 로그 기록 (loss가 None이 아니라는 가정 하에)
        self.log("loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def validation_step(self, batch, batch_idx):
        if self.use_additional_obj:
            feats, labels, length = batch
            pouts, loss = self(feats, length=length)
            for i, logit in enumerate(pouts):
                label = labels[:, i].reshape(-1)
                loss += self.criterion(logit, label)
        else:
            feats, length = batch
            loss = self(feats, length=length)
        validdict = {
            'valid_loss': loss,
        }
        self.log_dict(validdict, on_epoch=True, logger=True, sync_dist=self.distributed)
        return loss