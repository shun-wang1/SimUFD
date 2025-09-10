from model.model_base import *

class Encoder(nn.Module):
    def __init__(self, latent_dim=64, input_channels=1, flattened_dim=128):
        super(Encoder, self).__init__()
        conv_configs = [
            # channels_out, kernel_size, stride, padding
            (16, 64, 8, 31),  # layer1
            (32, 3, 1, 1),  # layer2
            (64, 3, 1, 1),  # layer3
            (64, 3, 1, 1),  # layer4
            (64, 3, 1, 1),  # layer5
            (64, 3, 1, 1),  # layer6
        ]

        layers = []
        in_channels = input_channels

        for i, (out_channels, kernel_size, stride, padding) in enumerate(conv_configs):
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),
                nn.MaxPool1d(2, stride=2)
            ])
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.flatten = nn.Flatten()

        self.flattened_dim = flattened_dim
        self.fc_mean = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = self.flatten(x)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

class Decoder(nn.Module):
    def __init__(self, latent_dim=64, final_out_channels=1, flattened_dim=128):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, flattened_dim),  #
            nn.Dropout(0.1)
        )
        self.unflatten = nn.Unflatten(1, (64, 2))
        deconv_configs = [
            # channels_out, kernel_size, stride, padding
            (64, 3, 1, 1),  # layer1
            (64, 3, 1, 1),  # layer2
            (64, 3, 1, 1),  # layer3
            (32, 3, 1, 1),  # layer4
            (16, 3, 1, 1),  # layer5
        ]
        layers = []
        in_channels = 64

        for out_channels, kernel_size, stride, padding in deconv_configs:
            layers.extend([
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(),
            ])
            in_channels = out_channels

        self.decoder_layers = nn.Sequential(*layers)
        self.output_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(16, final_out_channels, 64, stride=8, padding=28)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.decoder_layers(x)
        x = self.output_layer(x)
        x = x.squeeze(1)
        return x

class Model_VAE(nn.Module):
    def __init__(self, latent_dim, flattened_dim):
        super(Model_VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=latent_dim, flattened_dim=flattened_dim)
        self.decoder = Decoder(latent_dim=latent_dim, flattened_dim=flattened_dim)

    def forward(self, x):
        mean, logvar = self.encoder(x)
        z = self.encoder.reparameterize(mean, logvar)
        decoded = self.decoder(z)
        return decoded, mean, logvar

    def encode(self, x):
        mean, logvar = self.encoder(x)
        return self.encoder.reparameterize(mean, logvar)

    def decode(self, z):
        return self.decoder(z)

class VAETrainer(ModelTrainer):
    """Trainer for Variational Autoencoder"""
    def __init__(self, model, args, device):
        super().__init__(model, args, device)
        self.vae_recon_weight = args.vae_recon_weight
        self.vae_kl_weight = args.vae_kl_weight

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        recon_loss_total = 0
        kl_loss_total = 0

        for batch_idx, batch in enumerate(train_loader):
            data = batch[1].to(self.device)
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar = self.model(data)

            # Calculate loss components
            recon_loss = F.mse_loss(recon_batch, data, reduction='sum')/ data.size(0)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1), dim = 0)

            # Combine losses
            loss = self.vae_recon_weight * recon_loss + self.vae_kl_weight * kl_loss

            # Backward and optimize
            loss.backward()

            # Gradient clipping
            if self.args.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip_norm)

            optimizer.step()

            total_loss += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()

        dataloader_len = len(train_loader)
        avg_total_loss = total_loss / dataloader_len
        avg_recon_loss = recon_loss_total / dataloader_len
        avg_kl_loss = kl_loss_total / dataloader_len

        additional_metrics = {
            'Recon Loss': avg_recon_loss,
            'KL Loss': avg_kl_loss
        }

        return avg_total_loss, additional_metrics

    def validate_epoch(self, valid_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_loader):
                data = batch[1].to(self.device)

                # Forward pass
                recon_batch, mu, logvar = self.model(data)

                # Calculate loss components
                recon_loss = F.mse_loss(recon_batch, data, reduction='sum')/ data.size(0)
                kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)

                # Combine losses
                loss = self.vae_recon_weight * recon_loss + self.vae_kl_weight * kl_loss

                total_loss += loss.item()

        dataloader_len = len(valid_loader)
        avg_total_loss = total_loss / dataloader_len

        return avg_total_loss

    def plot_training_history(self, train_losses, valid_losses, additional_metrics_history=None):
        # Basic data with train and validation losses
        data = {
            'epoch': list(range(1, len(train_losses) + 1)),
            'train_loss': train_losses,
            'valid_loss': valid_losses
        }

        # Add additional metrics if provided
        if additional_metrics_history is not None:
            for metric_name, metric_values in additional_metrics_history.items():
                data[metric_name.lower().replace(' ', '_')] = metric_values

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.args.save_path, f'losses_{self.args.model}_{self.args.data_type}.csv')
        df.to_csv(csv_path, index=False)
        print(f"Losses saved to {csv_path}")

        # Create a simple plot for quick reference
        plt.style.use('seaborn-whitegrid')
        plt.figure(figsize=(8, 6))
        plt.plot(data['epoch'], train_losses, 'b-', label='Training Loss')
        plt.plot(data['epoch'], valid_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'{self.args.model} Training History')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.args.save_path, f'{self.args.model}_training_history.png'), dpi=300)
        plt.close()

    def calculate_threshold(self, train_loader, valid_loader):
        self.load_checkpoint()
        self.model.eval()

        with torch.no_grad():
            _, train_errors = self.collect_predictions(train_loader)
            _, val_errors = self.collect_predictions(valid_loader)

        threshold_errors = np.concatenate([train_errors, val_errors])
        threshold = self.calculate_threshold_rule(threshold_errors, method=self.args.threshold_method)

        return threshold

    def collect_predictions(self, data_loader):
        self.model.eval()
        all_errors = defaultdict(list)

        sample_times = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch[1].to(self.device)
                _ = self.model(data)
                break

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                data = batch[1].to(self.device)
                timestep = batch[0]

                pred, _, _ = self.model(data)
                # errors, _, _ = self.model(data)        # torch.mean((pred - input) ** 2, dim=1)
                errors = torch.mean((pred - data) ** 2, dim=1)

                for ts, err in zip(timestep.cpu().numpy(), errors.cpu().numpy()):
                    all_errors[int(ts)].append(err)

        averaged_errors = {ts: np.mean(err) for ts, err in all_errors.items()}

        timesteps = list(averaged_errors.keys())
        average_hi = list(averaged_errors.values())

        return timesteps, average_hi

    def validate_and_visualize(self, all_loader, threshold):

        self.model.eval()

        with torch.no_grad():
            all_timesteps, all_errors = self.collect_predictions(all_loader)

            train_end = self.args.train_timesteps
            val_end = self.args.val_timesteps

            train_results = (all_timesteps[:train_end], all_errors[:train_end])
            val_results = (all_timesteps[train_end:val_end], all_errors[train_end:val_end])
            test_results = (all_timesteps[val_end:], all_errors[val_end:])

            self.plot_health_index(train_results, val_results, test_results, threshold)

            results_dict = {
                'timestep': all_timesteps,
                'HI_score': all_errors,
                'dataset': ['train'] * train_end +
                           ['valid'] * (val_end - train_end) +
                           ['test'] * (len(all_timesteps) - val_end)
            }

        df = pd.DataFrame(results_dict)
        csv_path = os.path.join(self.args.save_path, f'HI_{self.args.model}_{self.args.data_type}.csv')
        df.to_csv(csv_path, index=False)
        return train_results, val_results, test_results

