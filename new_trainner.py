
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import check_gpu_status, get_logger, get_dataloader
from dataset import EEGDataset
from dataclasses import dataclass
from tqdm import tqdm

logger = get_logger(__name__)

class Trainer:
	def __init__(self, cfg, model, dataloader, optimizer=None, criterion=None, scheduler=None, writer=None):
		self.cfg = cfg
		self.device = torch.device(self.cfg.device_ids[0] if torch.cuda.is_available() else "cpu")
		self.model = model.to(self.device)
		self.optimizer = optimizer if optimizer else self._init_optimizer()
		self.criterion = criterion if criterion else self._init_criterion()
		self.scheduler = scheduler if scheduler else self._init_scheduler()
		self.train_loader, self.test_loader = dataloader[0], dataloader[1]
		self.early_stop_counter = 0
		self.best_acc = 0.0
		
	def _init_criterion(self):
		if self.cfg.criterion == "cross_entropy":
			return nn.CrossEntropyLoss()
		else:
			raise NotImplementedError("Criterion not implemented")
			
	def _init_optimizer(self):
		return optim.Adam(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
	
	def _init_scheduler(self):
		return CosineAnnealingLR(self.optimizer, T_max=self.cfg.epochs, eta_min=0)
	
	def train_one_epoch(self, epoch):
		self.model.train()
		total_loss = 0.0
		for data, target in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
			data, target = data.to(self.device), target.to(self.device)
			self.optimizer.zero_grad()
			output = self.model(data)
			loss = self.criterion(output, target)
			loss.backward()
			self.optimizer.step()
			total_loss += loss.item()
		avg_loss = total_loss / len(self.train_loader)
		return avg_loss
	
	def test_one_epoch(self):
		self.model.eval()
		total_loss = 0.0
		total_correct = 0
		with torch.no_grad():
			for data, target in self.test_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = self.model(data)
				loss = self.criterion(output, target)
				pred = output.argmax(dim=1)
				total_correct += pred.eq(target).sum().item()
				total_loss += loss.item()
		avg_loss = total_loss / len(self.test_loader)
		avg_acc = total_correct / len(self.test_loader.dataset)
		return avg_loss, avg_acc
	
	def train(self):
		for epoch in range(1, self.cfg.epochs + 1):
			train_loss = self.train_one_epoch(epoch)
			test_loss, test_acc = self.test_one_epoch()
			logger.info(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Test Loss {test_loss:.4f}, Accuracy {test_acc:.2f}")
			if self.early_stop(test_acc):
				logger.info("Early stopping triggered.")
				break
			self.scheduler.step()
			
	def early_stop(self, test_acc):
		if test_acc > self.best_acc:
			self.best_acc = test_acc
			self.early_stop_counter = 0
			self.save_best_model() # Save the best model on achieving higher accuracy
		else:
			self.early_stop_counter += 1
		if self.early_stop_counter >= self.cfg.early_stop_patience:
			logger.info(f"Early stopping triggered after {self.early_stop_counter} epochs without improvement.")
			return True
		return False
	
	def save_best_model(self):
		"""Saves the current best model based on accuracy."""
		checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.model_name}_best.pth")
		os.makedirs(self.cfg.checkpoint_dir, exist_ok=True)
		torch.save({
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'scheduler_state_dict': self.scheduler.state_dict(),
			'epoch': self.cfg.epochs,
			'best_acc': self.best_acc,
		}, checkpoint_path)
		logger.info(f"Best model saved with accuracy {self.best_acc:.2f} at {checkpoint_path}")
		
	def resume(self, checkpoint_path=None):
		"""Resumes training from a saved checkpoint."""
		if checkpoint_path is None:
			checkpoint_path = os.path.join(self.cfg.checkpoint_dir, f"{self.cfg.model_name}_best.pth")
		checkpoint = torch.load(checkpoint_path, map_location=self.device)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		self.cfg.epochs += checkpoint['epoch']
		self.best_acc = checkpoint['best_acc']
		logger.info(f"Resumed training from epoch {checkpoint['epoch']} with best accuracy {self.best_acc:.2f}")