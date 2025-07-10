import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from dino_sequence_dataset import DinoSequenceDataset  
from sequence_transformer import ASDTransformer  
from dino_finetune_main import my_collate_fn 

def test_model(model, test_loader, device, criterion_decision, criterion_action):
    model.eval()
    test_loss_total = 0
    test_decision_correct = 0
    test_decision_total = 0
    test_action_error = 0
    all_decisions = []
    all_decision_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            s_a = batch['s_a'].to(device)
            s_s = batch['s_s'].to(device)
            s_d = batch['s_d'].to(device)

            s_a_idx = batch['s_a_idx'].to(device)
            s_s_idx = batch['s_s_idx'].to(device)
            s_d_idx = batch['s_d_idx'].to(device)

            s_a_idx_batch = batch['s_a_idx_batch'].to(device)
            s_s_idx_batch = batch['s_s_idx_batch'].to(device)
            s_d_idx_batch = batch['s_d_idx_batch'].to(device)

            student_mask = batch['student_mask'].to(device).bool()
            target_d = batch['target_d'].to(device)
            target_a = batch['target_a'].to(device).float()

            decision_logits, action_pred = model(s_a, s_s, s_d,
                                            s_a_idx, s_s_idx, s_d_idx, 
                                            student_mask,
                                            s_a_idx_batch, s_s_idx_batch, s_d_idx_batch,
                                            mask_d=args.maskd, 
                                            finetune_type=args.finetune_type)
            d_loss = criterion_decision(decision_logits, target_d.squeeze(-1))
            a_loss = criterion_action(action_pred,target_a)
            loss = d_loss + a_loss

            test_loss_total += loss.item()

            _, decision_pred = torch.max(decision_logits, 1)
            test_decision_correct += (decision_pred == target_d.squeeze(-1)).sum().item()
            test_decision_total += target_d.size(0)

            test_action_error += torch.abs(action_pred - target_a).sum().item()

            all_decisions.extend(target_d.squeeze(-1).cpu().numpy())
            all_decision_preds.extend(decision_pred.cpu().numpy())
            

    avg_loss = test_loss_total / len(test_loader)
    acc = test_decision_correct / test_decision_total
    mae = test_action_error / test_decision_total

    print(f"Test Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}, Test MAE: {mae:.4f}")
    return avg_loss, acc, mae, all_decisions, all_decision_preds


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ASDTransformer(mode="finetune").to(device)
    model.load_state_dict(torch.load(args.weight_path, map_location=device))

    test_dataset = DinoSequenceDataset(args.test_data_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=my_collate_fn)

    criterion_decision = nn.CrossEntropyLoss()
    criterion_action = nn.L1Loss()

    test_loss, test_acc, test_mae, y_true, y_pred = test_model(model, test_loader, device, criterion_decision, criterion_action)

    time_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)/ f"finetune_test_{time_tag}"
    figures_dir = output_dir / f"figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame({
        'test_loss': [test_loss],
        'test_acc': [test_acc],
        'test_mae': [test_mae]
    })
    metrics_df.to_csv(output_dir / f"test_metrics.csv", index=False)

    plt.figure()
    plt.bar(['Loss', 'Acc', 'MAE'], [test_loss, test_acc, test_mae])
    plt.title("Final Test Metrics")
    plt.savefig(figures_dir / "test_bar_metrics.png")
    plt.close()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Test Confusion Matrix")
    plt.savefig(figures_dir / "test_confusion_matrix.png")
    plt.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to finetuned model weights')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--train_mode', type=str, choices=['linear', 'finetune'], default='finetune',
                        help="Training mode: linear (linear probing) or finetune (full fine-tuning)")
    parser.add_argument('--finetune_type', type=int, default=0, choices=[0, 1,2,3]) # 0: 全局 cls, 1: 最后几个 A/S
    parser.add_argument("--maskd", type=bool, default=False)
    args = parser.parse_args(
        [
            '--test_data_path', '../dino_data/dino_sequence_data/finetune_test.pt',
            '--weight_path', '../dino_data/output_dino/finetune_masked/weights/student_finetune_epoch20.pth',
            '--output_dir', '../dino_data/output_dino',
            '--finetune_type', '0',
            '--maskd', 'True',
            '--train_mode', 'finetune',
        ]
    )
    main(args)
