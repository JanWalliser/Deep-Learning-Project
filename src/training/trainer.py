import torch
from torch.nn import functional as F

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        n += x.size(0)
    return {"loss": total_loss / n, "acc": correct / n}

def train(model, train_loader, val_loader, device, epochs, lr, weight_decay, wandb_run=None):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(lr), weight_decay=weight_decay)

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()

            if wandb_run is not None:
                wandb_run.log({"train/loss": loss.item(), "step": global_step, "epoch": epoch})
            global_step += 1

        val_metrics = evaluate(model, val_loader, device)
        if wandb_run is not None:
            wandb_run.log({f"val/{k}": v for k, v in val_metrics.items()} | {"epoch": epoch})

    return model
