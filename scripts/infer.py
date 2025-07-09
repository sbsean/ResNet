import torch
import json
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
import numpy as np

# 현재 스크립트의 위치를 기준으로 상대 경로 계산하여 모듈 경로 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)  # 부모 디렉토리를 Python 경로에 추가

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder

from model.ResNet50 import ResNet50


def main():
    print("ResNet50 for CIFAR10 evaluation")

    if len(sys.argv) >= 2:
        config_filename = sys.argv[1]
        print("Config file:", config_filename)
    else:
        
        config_filename = os.path.join(parent_dir, "config", "config.json")

   
    if not os.path.exists(config_filename):
        print(f"Config file not found: {config_filename}")
        
        params = {"task": "CIFAR10"}
        print("Using default CIFAR10 configuration")
    else:
        print(f"Loading config from: {config_filename}")
        with open(config_filename, "r", encoding="UTF8") as f:
            params = json.load(f)

    
    print("GPU Available:", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    
    checkpoint_dir = None
    
    
    if len(sys.argv) >= 3:
        checkpoint_dir = sys.argv[2]
        print(f"Using specified checkpoint: {checkpoint_dir}")
    
   
    if checkpoint_dir is None:
        print("\n" + "="*60)
        print("CHECKPOINT SELECTION")
        print("="*60)
        
       
        checkpoints_base = os.path.join(parent_dir, "checkpoints")
        available_checkpoints = []
        
        if os.path.exists(checkpoints_base):
            
            setting_dirs = [d for d in os.listdir(checkpoints_base) if d.startswith("setting_")]
            setting_dirs.sort(key=lambda x: int(x.split("#")[1]))
            
            for setting_dir in setting_dirs:
                ckpt_path = os.path.join(checkpoints_base, setting_dir, "ckpt")
                if os.path.exists(ckpt_path):
                    
                    for ckpt_file in ["best_model.pth", "final_model.pth"]:
                        full_path = os.path.join(ckpt_path, ckpt_file)
                        if os.path.exists(full_path):
                            available_checkpoints.append({
                                'path': full_path,
                                'setting': setting_dir,
                                'type': ckpt_file.replace('.pth', ''),
                                'display': f"{setting_dir}/{ckpt_file}"
                            })
        
        if not available_checkpoints:
            print("❌ No checkpoints found!")
            print(f"Expected location: {checkpoints_base}")
            print("Please train a model first using main.py")
            sys.exit(1)
        
        
        print("Available checkpoints:")
        print("-" * 40)
        for i, ckpt in enumerate(available_checkpoints):
            print(f"{i+1:2d}. {ckpt['display']}")
        
        print(f"{len(available_checkpoints)+1:2d}. Enter custom path")
        print("-" * 40)
        
       
        while True:
            try:
                choice = input(f"Select checkpoint (1-{len(available_checkpoints)+1}): ").strip()
                
                if choice == str(len(available_checkpoints)+1):
                   
                    custom_path = input("Enter checkpoint path: ").strip()
                    if os.path.exists(custom_path):
                        checkpoint_dir = custom_path
                        print(f"✅ Using custom checkpoint: {checkpoint_dir}")
                        break
                    else:
                        print(f"❌ File not found: {custom_path}")
                        continue
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(available_checkpoints):
                    checkpoint_dir = available_checkpoints[choice_idx]['path']
                    print(f"✅ Selected: {available_checkpoints[choice_idx]['display']}")
                    break
                else:
                    print(f"❌ Invalid choice. Please enter 1-{len(available_checkpoints)+1}")
                    
            except ValueError:
                print(f"❌ Invalid input. Please enter a number 1-{len(available_checkpoints)+1}")
            except KeyboardInterrupt:
                print("\n❌ Cancelled by user")
                sys.exit(0)
        
        print("="*60)

    # 데이터 변환 설정
    if params.get("task", "CIFAR10") == "CIFAR10":
        transforms_test = transforms.Compose([
            transforms.Resize([32, 32]),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    # example 디렉토리 경로 사용
    example_dir = os.path.join(current_dir, "example")

    imgs = ImageFolder(example_dir, transform=transforms_test)
    print("Total images found:", len(imgs))
    
    if len(imgs) == 0:
        print("No images found! Please check the directory structure.")
        print("Make sure images are in subdirectories within 'example' folder.")
        sys.exit(1)
    
    inference_loader = torch.utils.data.DataLoader(imgs, batch_size=1)

    # ResNet50 모델 생성
    model = ResNet50().to(device)
    print("ResNet50 model created")

    # validation mode
    model.eval()

    try:
        checkpoint = torch.load(checkpoint_dir, map_location=device)
        
        # 체크포인트 정보 표시
        print("\n" + "="*50)
        print("CHECKPOINT INFORMATION")
        print("="*50)
        
        # 체크포인트 구조 확인
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"✅ Model loaded successfully!")
            
            # 추가 정보 표시
            if "epoch" in checkpoint:
                print(f"📅 Epoch: {checkpoint['epoch']}")
            if "train_loss" in checkpoint:
                print(f"📉 Training Loss: {checkpoint['train_loss']:.4f}")
            if "train_acc" in checkpoint:
                print(f"📊 Training Accuracy: {checkpoint['train_acc']:.2f}%")
            if "val_loss" in checkpoint:
                print(f"📉 Validation Loss: {checkpoint['val_loss']:.4f}")
            if "val_acc" in checkpoint:
                print(f"🎯 Validation Accuracy: {checkpoint['val_acc']:.2f}%")
        else:
           
            model.load_state_dict(checkpoint)
            print("✅ Model state loaded successfully!")
            
        print("="*50)
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)

    # CIFAR-10 클래스 
    classes = (
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    total_images = len(inference_loader)
    print(f"Processing {total_images} images...")

    
    cols = min(5, total_images)  
    rows = (total_images + cols - 1) // cols  

    plt.figure(figsize=(cols * 3, rows * 3))  

    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():  
        for i, (thisimg, label_batch) in enumerate(inference_loader):
            
            img = thisimg[0]
            label = label_batch.item()  

            
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
            img_display = img * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1)  

            
            pred = model(thisimg.to(device))
            softmax = torch.nn.functional.softmax(pred, dim=1)
            confidence, top_pred = torch.max(softmax, dim=1)
            confidence = confidence.item() * 100  
            top_pred = top_pred.item()

            # 정확도 계산
            if label == top_pred:
                correct_predictions += 1
            total_predictions += 1

            
            ax = plt.subplot(rows, cols, i + 1)
            plt.imshow(img_display)
            plt.axis("off") 

          
            true_class = classes[label]
            pred_class = classes[top_pred]

            color = "green" if label == top_pred else "red"
            
            
            if label == top_pred:
                display_text = f"✓ {pred_class}\n{confidence:.1f}%"
            else:
                display_text = f"✗ {pred_class}\n(True: {true_class})\n{confidence:.1f}%"

            ax.text(
                0.5, -0.15,
                display_text,
                horizontalalignment="center",
                verticalalignment="top",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
            )

          
            print("=" * 50)
            print(f"Image {i+1}: {os.path.basename(imgs.imgs[i][0])}")
            print(f"True class: {true_class}")
            print(f"Prediction: {pred_class} (Confidence: {confidence:.2f}%)")
            print(f"Correct: {'✓' if label == top_pred else '✗'}")

    plt.tight_layout()

    # accuracy
    accuracy = (correct_predictions / total_predictions) * 100
    print("\n" + "=" * 50)
    print(f"FINAL RESULTS:")
    print(f"Total images: {total_predictions}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 50)

    
    result_path = os.path.join(current_dir, "resnet50_prediction_results.png")
    plt.savefig(result_path, dpi=150, bbox_inches='tight')
    
    try:
        plt.show()
    except:
        print("Cannot display plot (no GUI), but image saved successfully.")
    
    print(f"\nResults saved to: {result_path}")

    # 결과 요약을 텍스트 파일로도 저장
    summary_path = os.path.join(current_dir, "resnet50_inference_summary.txt")
    with open(summary_path, "w") as f:
        f.write("ResNet50 Inference Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Model: ResNet50\n")
        f.write(f"Checkpoint: {checkpoint_dir}\n")
        f.write(f"Total images: {total_predictions}\n")
        f.write(f"Correct predictions: {correct_predictions}\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n\n")
        f.write("Individual Results:\n")
        f.write("-" * 30 + "\n")
        
        for i, (filename, _) in enumerate(imgs.imgs):
            img_name = os.path.basename(filename)
            f.write(f"{i+1}. {img_name}\n")
    
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()