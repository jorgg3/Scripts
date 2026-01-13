import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models, transforms, datasets
import lightning as L


class CowResNet(L.LightningModule):
    """
    Modelo:
      - backbone ResNet50 
      - training_step / validation_step
    """
    def __init__(self, lr=0.0001):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Cargamos ResNet50 preentrenada
        backbone = models.resnet50(weights="DEFAULT")

        # Cómo y porqué trabajamos con la última capa?...Me parece que en resnet50 esto es lo que clasifica 
        capas = list(backbone.children())
        self.feature_extractor = nn.Sequential(*capas[:-1])
        num_filters = backbone.fc.in_features 

        #Aquí van los pesos y el bias 
        self.classifier = nn.Linear(num_filters, NUM_CLASSES)

        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x): #Falta definir batch size
        # backbone -> [batch size, 2048, 1, 1]
        features = self.feature_extractor(x)
        # flatten -> [batch size, 2048]
        features = features.flatten(1)
        # classifier -> [B, 2], se muestras los pesos
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        #precisión
        acc = (preds == y).float().mean()
        #Esto solo para verlo mientras se ejecuta
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        # Optimizador, el más común es Adam, supongo que usaremos ese 
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def main():
    dm = #Aquí debería cargar el dataset, especificando el batchsize, el número de epochs, cómo se hacer el slicing, etc...
    model = CowResNet()
    trainer = L.Trainer(
        max_epochs=#máximo de epochs,
        accelerator=#Deberiamos poder usar la GPU,
        devices=#Lo mismo que antes, usar la GPU,
        callbacks= #Funciones adicionales, aún no se bien cuáles SI tengo que implementar ,
        log_every_n_steps=10 #Para no saturar la consola 
    )

    print("Entrenanod")
    trainer.fit(model, datamodule=dm)



if __name__ == "__main__":
    main()
