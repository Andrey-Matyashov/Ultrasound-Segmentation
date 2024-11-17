class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        optim,
        loss_fn,
        num_epoch = 2
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optim = optim
        self.loss_fn = loss_fn
        self.num_epochs = num_epoch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
    @staticmethod
    def dice(net_outp, gt):
        
        y_pred = net_outp.argmax(dim = 1).squeeze()
        y_true = gt.squeeze()
        intersection = (y_true * y_pred).sum()
        dice = (2.0 * intersection) / (y_true.sum() + y_pred.sum())
        return dice
        
    def training(self):
        
        train_losses = []
        train_dice_metrics = []
        test_losses = []
        test_dice_metrics = []
        
        for epoch in tqdm(range(self.num_epochs)):
            gc.collect()
            
            print(f'####### epoch = {epoch} ########')
            
            train_losses_on_epoch = []
            train_dice_metrics_on_epoch = []
            
            for x, y in tqdm(self.train_loader):
                if (x is None) or (y is None):
                    continue
                y =  y.to(torch.int64)
                output = self.model(x.to(self.device)).cpu()
                loss = self.loss_fn(output, y.squeeze())
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                
                train_losses_on_epoch.append(loss.item())
                train_dice_metrics_on_epoch.append(self.dice(output, y).item())
                
            test_losses_on_epoch = []
            test_dice_metrics_on_epoch = []
            
            with torch.no_grad():
                for x, y in tqdm(self.test_loader):
                    if (x is None) or (y is None):
                        continue
                    y =  y.to(torch.int64)
                    output = self.model(x.to(self.device)).cpu()
                    loss = self.loss_fn(output, y.squeeze())
                    
                    test_losses_on_epoch.append(loss.item())
                    test_dice_metrics_on_epoch.append(self.dice(output, y).item())
                    
            train_losses.append(np.mean(train_losses_on_epoch))
            train_dice_metrics.append(np.mean(train_dice_metrics_on_epoch))
            test_losses.append(np.mean(test_losses_on_epoch))
            test_dice_metrics.append(np.mean(test_dice_metrics_on_epoch))
            
            clear_output()
            fig, ax = plt.subplots(1, 2)
            ax[0].plot(range(len(train_losses)), train_losses, label='losses on train')
            ax[1].plot(range(len(train_dice_metrics)), train_dice_metrics, label = 'train dice metric')
            ax[0].plot(range(len(test_losses)), test_losses, label='losses on test')
            ax[1].plot(range(len(test_dice_metrics)), test_dice_metrics, label = 'test dice metric')

            plt.grid()
            plt.show()
                
                