import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .dataloader import PretrainEmoDataset, SecondPhaseEmoDataset, RandomBucketSampler, StandardSampler, BaselineDataset
from torch.utils import data
from modules.FeatureFuser import Wav2vecWrapper, Wav2vec2Wrapper, Wav2vec2PretrainWrapper
from modules.NN import LinearHead, RNNLayer
from utils.metrics import ConfusionMetrics
import pytorch_lightning as pl

class PretrainedEmoClassifier(pl.LightningModule):
    def __init__(self, max_epochs, batch_size, lr, datadir, labeldir, modelpath, labeling_method, valid_split):
        super().__init__()
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.wav2vec = Wav2vecWrapper(modelpath)
        self.data = PretrainEmoDataset(datadir, labeldir, labeling_method=labeling_method)
        self.linearhead = LinearHead(512, 128, self.data.nemos)
        self.linearhead_rnn = LinearHead(512, 128, self.data.nemos)
        self.rnn = RNNLayer(512, 512)
        numtraining = int(len(self.data) * valid_split)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))
        counter = self.data.emos
        weights = torch.tensor(
            [counter[c] for c in self.data.emoset]
        ).float()
        # detach arithmetic results so that the resulting tensor is a leaf
        weights = (weights.sum() / weights).detach()
        weights = (weights / weights.sum()).detach()
        print(f"Weigh losses by prior distribution of each class: {weights}.")
        self.criterion = nn.CrossEntropyLoss(weight=weights)

    def forward(self, x):
        reps = self.wav2vec(x)
        logits_reduced = self.linearhead_rnn(self.rnn(reps))
        logits = self.linearhead(reps)
        return logits, logits_reduced

    def train_dataloader(self):
        return data.DataLoader(self.traindata,
                               batch_size=self.batch_size,
                               num_workers=0,
                               drop_last=True,
                               shuffle=True,
                               collate_fn=self.data.truncCollate)

    def val_dataloader(self):
        return data.DataLoader(self.valdata,
                               batch_size=self.batch_size,
                               num_workers=0,
                               drop_last=False,
                               shuffle=False,
                               collate_fn=self.data.truncCollate)

    def configure_optimizers(self):
        parameters = list(self.linearhead.parameters()) + list(self.wav2vec.trainable_params()) + \
                     list(self.rnn.parameters()) + list(self.linearhead_rnn.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr)
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
        pout, pout_reduced = self(feats)
        loss = self.criterion(pout_reduced, label)
        label = label.unsqueeze(1).expand(-1, pout.size(1)).reshape(-1)
        pout = pout.reshape(-1, pout.size(2))
        loss += self.criterion(pout, label)
        acc = (label == torch.argmax(pout, -1)).float().mean()
        tqdm_dict = {'loss': loss, 'acc': acc}
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
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
                               num_workers=0,
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
                               num_workers=0,
                               batch_sampler=sampler,
                               collate_fn=self.data.seqCollate)

    def configure_optimizers(self):
        optimizer = optim.Adam(list(self.linearheads.parameters()) + list(self.wav2vec2.trainable_params()), lr=self.lr)
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

    def training_step(self, batch, batch_idx):
        feats, labels, length = batch
        pouts, maskidx = self(feats, length=length)
        loss, acc = 0, 0
        for i, logit in enumerate(pouts):
            label = labels[:, i][maskidx]
            loss += self.criterion(logit, label)
            acc += (label == torch.argmax(logit, -1)).float().mean()
        acc /= len(pouts)
        tqdm_dict = {'loss': loss, 'acc': acc}
        self.log_dict(tqdm_dict, on_step=True, on_epoch=True, prog_bar=True)
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
            nn.Linear(512, n_classes)
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
        self.save_hyperparameters()

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
            self.linearheads = nn.ModuleList(
                [nn.Linear(self.wav2vec2.wav2vec2PT.config.proj_codevector_dim, ncluster) for i, ncluster in enumerate(nclusters)]
            )
            self.data = SecondPhaseEmoDataset(
                datadir, None, labelpath, maxseqlen=maxseqlen,
                final_length_fn=self.wav2vec2.get_feat_extract_output_lengths
            )
            self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        else:
            self.data = BaselineDataset(datadir, labelpath, maxseqlen=maxseqlen)

        numtraining = int(len(self.data) * 0.9)
        splits = [numtraining, len(self.data) - numtraining]
        self.traindata, self.valdata = data.random_split(self.data, splits, generator=torch.Generator().manual_seed(58))

        print(f"[ContinueFinetuningBaseline] Dataset split 완료 - Train: {len(self.traindata)}, Val: {len(self.valdata)}")

    def train_dataloader(self):
        print("[ContinueFinetuningBaseline] train_dataloader 생성")
        idxs = self.traindata.indices
        length = None
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [min(l, self.maxseqlen) for l in length]

        sampler = RandomBucketSampler(self.train_bucket_size, length, self.batch_size, drop_last=True,
                                      distributed=self.distributed, world_size=self.trainer.world_size,
                                      rank=self.trainer.local_rank) if self.use_bucket_sampler else \
                  data.BatchSampler(
                      StandardSampler(self.traindata, shuffle=True, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank),
                      self.batch_size, drop_last=True
                  )

        loader = data.DataLoader(self.traindata,
                                 num_workers=4,
                                 batch_sampler=sampler,
                                 collate_fn=self.data.seqCollate)

        self._debug_sample_batch(loader, "Train")
        return loader

    def val_dataloader(self):
        print("[ContinueFinetuningBaseline] val_dataloader 생성")
        idxs = self.valdata.indices
        length = None
        if self.use_bucket_sampler:
            length = [self.data.lengths[i] for i in idxs]
            length = [min(l, self.maxseqlen) for l in length]

        sampler = RandomBucketSampler(self.val_bucket_size, length, self.batch_size, drop_last=True,
                                      distributed=self.distributed, world_size=self.trainer.world_size,
                                      rank=self.trainer.local_rank) if self.use_bucket_sampler else \
                  data.BatchSampler(
                      StandardSampler(self.valdata, shuffle=False, distributed=self.distributed,
                                      world_size=self.trainer.world_size, rank=self.trainer.local_rank),
                      self.batch_size, drop_last=False
                  )

        loader = data.DataLoader(self.valdata,
                                 num_workers=4,
                                 batch_sampler=sampler,
                                 collate_fn=self.data.seqCollate)

        self._debug_sample_batch(loader, "Validation")
        return loader

    def _debug_sample_batch(self, loader, phase):
        print(f"[{phase} DataLoader] 샘플 배치 로드 테스트")
        for batch in loader:
            if self.use_additional_obj:
                feats, labels, lengths = batch
                print(f"[{phase} 샘플] feats shape: {feats.shape}, labels shape: {labels.shape}, lengths: {lengths}")
            else:
                feats, lengths = batch
                print(f"[{phase} 샘플] feats shape: {feats.shape}, lengths: {lengths}")
            break

    def configure_optimizers(self):
        parameters = self.wav2vec2.trainable_params()
        if self.use_additional_obj:
            parameters += list(self.linearheads.parameters())

        optimizer = optim.Adam(parameters, lr=self.lr)
        num_training_steps = self.max_epochs
        num_warmup_epochs = self.warmup_epochs
        num_flat_steps = int(0.05 * num_training_steps)

        def lambda_lr(step):
            if step < num_warmup_epochs:
                return step / max(1, num_warmup_epochs)
            elif step < (num_warmup_epochs + num_flat_steps):
                return 1.0
            return max(0.0, float(num_training_steps - step) / float(num_training_steps - num_warmup_epochs - num_flat_steps))

        scheduler = {'scheduler': optim.lr_scheduler.LambdaLR(optimizer, lambda_lr), 'interval': 'epoch'}
        return [optimizer], [scheduler]

    def forward(self, x, length):
        if length is None:
            raise ValueError("[forward] length is None! 데이터로더에서 seqCollate가 제대로 작동했는지 확인 필요.")
        
        reps, pretrain_loss = self.wav2vec2(x, length)

        # pretrain_loss가 None이어도 TAPT 모드에서는 정상 동작
        if pretrain_loss is None:
            print("[INFO] Pretrain loss is None - this is expected during TAPT if no labels are used.")
            pretrain_loss = torch.tensor(0.0, device=x.device)  # 기본 0으로 둠 (로깅을 위해)

        return reps, pretrain_loss




    def training_step(self, batch, batch_idx):
        feats, lengths = batch

        # projected_states와 loss 둘 다 받기
        reps, loss = self.wav2vec2(feats, length=lengths)

        # projected_states로 추가적인 loss 계산할 경우 여기에 추가
        # loss += some_other_loss(reps)

        if loss is None:
            raise RuntimeError("[ERROR] Loss is None, check forward() output.")

        if not loss.requires_grad:
            raise RuntimeError("[ERROR] Loss does not require grad. Check the model’s forward method and ensure all parameters require grad.")

        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        feats, lengths = batch if not self.use_additional_obj else batch[:2]
        assert lengths is not None, "Validation Step - lengths is None!"

        reps, pretrain_loss = self(feats, length=lengths)

        self.log('valid_loss', pretrain_loss, prog_bar=True, on_epoch=True)

        return pretrain_loss  # 마찬가지로 loss만 반환

