import os
os.system('pip install pytorch-lightning')

import pytorch_lightning as pl

def create_model():
    
    '''
    The resnet34 model pretrained on Imagenet dataset is fetched from the torchvision
    models, and hence need to be tweaked slightly to suit the number of output classes.
    For that reason, the we replace the output classes in the final fc layer with 100 
    to suit the CIFAR100 dataset.
    '''

    model = torchvision.models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    num_classes = 100
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


class LitResnet34(pl.LightningModule):
    
    '''
    The class is a subset of pytorch-lightning's `LightningModule`. The `create_model`
    function defined above is included in the constructor. The forward call is made when
    we call the instance of the class with inputs which then returns the predictions.

    The methods `training_step`, `validation_step` and `test_step` are applicable for
    single run of the respective batch data. The `batch_idx` is applicable for
    `validation_step` and `test_step` to distinguish between validation and test steps.
    The method `evaluate` acts as template function for `validation_step` and `test_step`
    methods.
    '''

    def __init__(self, batch_size, learning_rate=1e-3):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model = create_model()

    # will be used during inference
    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    # logic for a single training step
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)

        # training metrics
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f'{stage}_loss', loss, prog_bar=True)
            self.log(f'{stage}_acc', acc, prog_bar=True)

        return loss

    # logic for a single validation step
    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    # logic for a single testing step
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")


    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9,weight_decay=5e-4,)
        steps_per_epoch = 45000 // self.batch_size
        scheduler_dict = { "scheduler": OneCycleLR(optimizer, 0.1, epochs=self.trainer.max_epochs, steps_per_epoch=steps_per_epoch,),
                            "interval": "step",}
        #return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
        return optimizer