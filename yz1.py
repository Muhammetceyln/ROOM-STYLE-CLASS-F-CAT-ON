import torch

# CUDA yoksa işlemciyi kullan
device = torch.device("cpu") 
print(f"Eğitim şu cihazda yapılacak: {device}")