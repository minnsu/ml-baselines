import torch
import random

class Trainer:
    def __init__(self, model, optimizer, criterion, mode):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        # 'classification', 'cl' or 'regression', 're'
        self.mode = mode

    def validate(self, dataloader):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, Y in dataloader:
                output = self.model(X)

                _, predicted = torch.max(output, 1)
                total += Y.size(0)
                correct += (predicted == Y).sum().item()
        
        return correct / total

    def train(self, epochs, dataloader, validloader, augment_transformer=None, augment_ratio=0, verbose=True, print_every=10):
        for epoch in range(1, epochs+1):
            for X, Y in dataloader:
                self.optimizer.zero_grad()
                
                if augment_transformer and augment_ratio > 0 and random.random() < augment_ratio:
                    X = augment_transformer(X)
                output = self.model(X)
                loss = self.criterion(output, Y)
                loss.backward()
                
                self.optimizer.step()
            
            if verbose:
                ratio = int(100 * epoch / epochs)
                if validloader is not None and self.mode == 'classification' or self.mode == 'cl':
                    accuracy = 100 * self.validate(validloader)
                    print(f'\rEpoch {epoch}[{ratio}%] Loss: {loss.item()} Accuracy: {accuracy:.2f}%', end='')
                else:
                    print(f'\rEpoch {epoch}[{ratio}%] Loss: {loss.item()}', end='')
                if epoch % print_every == 0:
                    print('')

    def test(self, test_dataloader):
        with torch.no_grad():
            ret = []
            for X, Y in test_dataloader:
                output = self.model(X)
                
                if self.mode == 'classification' or self.mode == 'cl':
                    _, predicted = torch.max(output, 1)
                    ret.extend(predicted.tolist())
                else:
                    ret.extend(output.tolist())
        return ret