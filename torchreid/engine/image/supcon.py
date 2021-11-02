from __future__ import division, print_function, absolute_import

from torchreid import metrics
from torchreid.losses import SupConLoss

from ..engine import Engine


class ImageSupConEngine(Engine):
    def __init__(
        self,
        datamanager,
        model,
        optimizer,
        scheduler=None,
        use_gpu=True,
    ):
        super(ImageSupConEngine, self).__init__(datamanager, use_gpu)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.register_model('model', model, optimizer, scheduler)

        self.criterion = SupConLoss(
            use_gpu=self.use_gpu,
        )

    def forward_backward(self, data):
        imgs, pids = self.parse_data_for_train(data)
    
        if self.use_gpu:
            imgs = imgs.cuda()
            pids = pids.cuda()

        if self.is_unsupervised:
            outputs, targets = self.model(imgs, imgs)
            print(outputs.shape)
            loss = self.criterion(features=outputs, labels=targets)
        else:
            outputs = self.model(imgs)
            loss = self.compute_loss(self.criterion, outputs, pids)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_summary = {
            'loss': loss.item(),
            'acc': metrics.accuracy(outputs, pids)[0].item()
        }

        return loss_summary
