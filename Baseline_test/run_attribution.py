import argparse
import os
import torch
import random
import glob
import sys
from tools import *
from ECAPAModel import ECAPAModel
from attribution.analyzer import ECAPAAttributionAnalyzer
from attribution.baseline import BaselineComputer

def get_samples(args):
    """
    获取待分析的音频样本
    """
    samples = []
    
    # 1. 优先使用命令行指定的样本
    if args.attribution_samples:
        samples = args.attribution_samples.split(',')
        print(f"[Attribution] Using specified samples: {samples}")
        return samples

    # 2. 如果未指定，尝试从eval_list随机选择样本
    print(f"[Attribution] No samples specified, trying to select from eval_list: {args.eval_list}")
    try:
        if os.path.exists(args.eval_list):
            with open(args.eval_list, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                # 随机选择几行
                selected_lines = random.sample(lines, min(len(lines), 20))
                candidates = []
                for line in selected_lines:
                    parts = line.strip().split()
                    # 通常第2和第3列是路径
                    if len(parts) >= 3:
                        p1 = os.path.join(args.eval_path, parts[1])
                        p2 = os.path.join(args.eval_path, parts[2])
                        if os.path.exists(p1): candidates.append(p1)
                        if os.path.exists(p2): candidates.append(p2)
                
                # 随机抽取10个数据
                if len(candidates) >= 1:
                    num_samples = min(len(candidates), 10)
                    samples = random.sample(candidates, num_samples)
                    print(f"[Attribution] Randomly selected {num_samples} samples from eval_list: {samples}")
                    return samples
    except Exception as e:
        print(f"[Attribution] Warning: Failed to read from eval_list: {e}")

    # 3. 如果eval_list失败，尝试从train_list选择
    print(f"[Attribution] Trying to select from train_list: {args.train_list}")
    try:
        if os.path.exists(args.train_list):
            with open(args.train_list, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 0:
                selected_lines = random.sample(lines, min(len(lines), 20))
                candidates = []
                for line in selected_lines:
                    parts = line.strip().split()
                    for part in parts:
                        full_path = os.path.join(args.train_path, part)
                        if os.path.exists(full_path):
                            candidates.append(full_path)
                            break
                        elif os.path.exists(part):
                            candidates.append(part)
                            break
                
                if len(candidates) >= 1:
                    num_samples = min(len(candidates), 10)
                    samples = random.sample(candidates, num_samples)
                    print(f"[Attribution] Randomly selected {num_samples} samples from train_list: {samples}")
                    return samples
    except Exception as e:
        print(f"[Attribution] Warning: Failed to read from train_list: {e}")

    # 4. 如果都失败，扫描目录
    print("[Attribution] Scanning directories for wav files...")
    wav_files = glob.glob(os.path.join(args.eval_path, "**/*.wav"), recursive=True)
    if not wav_files:
        wav_files = glob.glob(os.path.join(args.train_path, "**/*.wav"), recursive=True)
    
    if len(wav_files) >= 1:
        num_samples = min(len(wav_files), 10)
        samples = random.sample(wav_files, num_samples)
        print(f"[Attribution] Randomly scanned {num_samples} samples: {samples}")
        return samples

    print("[Attribution] Error: Could not find any audio samples.")
    return []

def main():
    parser = argparse.ArgumentParser(description = "ECAPA_attribution_comparison")
    
    # 必要参数 (与trainECAPAModel.py保持一致以便复用ECAPAModel初始化)
    parser.add_argument('--num_frames', type=int,   default=200,     help='Duration of the input segments')
    parser.add_argument('--lr',         type=float, default=0.001,   help='Learning rate')
    parser.add_argument("--lr_decay",   type=float, default=0.97,    help='Learning rate decay')
    parser.add_argument('--C',       type=int,   default=512,   help='Channel size for the speaker encoder')
    parser.add_argument('--m',       type=float, default=0.2,    help='Loss margin in AAM softmax')
    parser.add_argument('--s',       type=float, default=30,     help='Loss scale in AAM softmax')
    parser.add_argument('--n_class', type=int,   default=1211,   help='Number of speakers')
    parser.add_argument('--test_step',  type=int,   default=1,       help='Test and save every [test_step] epochs')
    parser.add_argument('--device',  type=int,   default=2,   help='device')

    # 模型路径参数 - 支持3个模型
    parser.add_argument('--initial_model_1', type=str,default="/home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_clean_noSpec/exp/vox1/model/model_0061.model", help='Path to model 1')
    parser.add_argument('--initial_model_2', type=str, default="/home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_noise_Spec/exp/vox1/model/model_0075.model", help='Path to model 2')
    parser.add_argument('--initial_model_3', type=str, default="/home/zhangxl24/SpeakerRecongnition/wode/Noise_adv_vox1/exps/3.05/model/model_0077.model", help='Path to model 3')
    # 其他路径参数
    parser.add_argument('--save_path',  type=str,   default="attribution_result", help='Path to save attribution results')
    parser.add_argument('--train_list', type=str, default="/home/zhangxl24/SpeakerRecongnition/somedata/train_list_1.txt")
    parser.add_argument('--train_path', type=str, default="/home/database/sre/voxceleb/voxceleb1/dev/wav")
    parser.add_argument('--eval_list', type=str, default="/home/database/sre/voxceleb/voxceleb1/voxceleb1_test_v2.txt")
    parser.add_argument('--eval_path', type=str, default="/home/database/sre/voxceleb/voxceleb1/test/wav")
    parser.add_argument('--musan_path', type=str, default="/home/database/noise/musan", help='The path to the MUSAN set')
    # 归因参数
    parser.add_argument('--attribution_samples', type=str, default="/home/database/sre/voxceleb/voxceleb1/test/wav/id10300/_WKW_Jkdvq8/00005.wav,/home/database/sre/voxceleb/voxceleb1/test/wav/id10270/GWXujl-xAVM/00004.wav,/home/database/sre/voxceleb/voxceleb1/test/wav/id10283/SwMoq9ZHxpw/00004.wav", help='Comma separated list of audio paths for attribution analysis')
    # 归因模式参数
    parser.add_argument('--mode', type=str, default='legacy',
                        choices=['legacy', 'paired'],
                        help='Attribution mode: legacy (original) or paired (positive/negative/difference)')
    parser.add_argument('--paired_list', type=str, default=None,
                        help='Path to paired sample list CSV file (for paired mode). Format: target_path,ref_same_path,ref_diff_path,label')
    parser.add_argument('--baseline_type', type=str, default='zero',
                        choices=['zero', 'global_mean', 'speaker_mean', 'cross_speaker_mean'],
                        help='Baseline type for IG')
    parser.add_argument('--objective', type=str, default='cosine_sim',
                        choices=['cosine_sim', 'l2_norm'],
                        help='Attribution objective function')

    args = parser.parse_args()

    # 确保保存目录存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 1. 加载三个模型
    models_paths = [args.initial_model_1, args.initial_model_2, args.initial_model_3]
    models_dict = {}
    model_identifiers = [] # 用于构建保存目录
    
    for i, path in enumerate(models_paths):
        print(f"[Attribution] Loading Model {i+1} from {path}...")
        if not os.path.exists(path):
            print(f"Error: Model file {path} not found.")
            sys.exit(1)
            
        try:
            # 实例化 ECAPAModel 并加载参数
            model = ECAPAModel(**vars(args))
            model.load_parameters(path)
            model.eval()
            
            # 提取模型标识符 (倒数第五级目录)
            try:
                path_parts = os.path.normpath(path).split(os.sep)
                epoch_str = os.path.splitext(os.path.basename(path))[0].split('_')[-1]
                
                if len(path_parts) >= 5:
                    dir_name = path_parts[-5]
                else:
                    dir_name = path_parts[-2] if len(path_parts) >= 2 else f"Model_{i+1}"
                
                # 构建用于显示的名称 (Key)
                # 如果名称已存在，追加后缀以防覆盖
                key_name = dir_name
                if key_name in models_dict:
                     key_name = f"{dir_name}_{epoch_str}" # 尝试加 epoch
                     if key_name in models_dict:
                         key_name = f"{dir_name}_{i+1}" # 还是重复，加序号
                
                # 存入字典，key_name 将被用作图表的列标题
                models_dict[key_name] = model.speaker_encoder
                
                # 保存标识符用于生成结果目录名
                model_identifiers.append(f"{dir_name}_{epoch_str}")
                
            except Exception as e:
                print(f"Error extracting name for model {path}: {e}")
                models_dict[f"Model_{i+1}"] = model.speaker_encoder
                model_identifiers.append(f"Model_{i+1}")

        except Exception as e:
            print(f"Error loading model {path}: {e}")
            sys.exit(1)

    samples = []
    sample_pairs = []

    if args.mode == 'legacy':
        # === Legacy Mode (original behavior) ===
        samples = get_samples(args)
        if not samples:
            print("[Attribution] Failed to get samples. Exiting.")
            return

        analyzer = ECAPAAttributionAnalyzer(
            models_dict=models_dict,
            C=args.C,
            device=args.device,
            musan_path=args.musan_path
        )

        subdir_name = "_vs_".join(model_identifiers)
        save_dir = os.path.join(args.save_path, subdir_name)
        print(f"[Attribution] Results will be saved to: {save_dir}")

        analyzer.analyze_and_save(samples, save_dir)
        print(f"[Attribution] Done! Results saved to {save_dir}")

    elif args.mode == 'paired':
        # === Paired Mode (positive/negative/difference attribution) ===
        if not args.paired_list or not os.path.exists(args.paired_list):
            print("[Attribution] Error: --paired_list is required for paired mode")
            return

        import csv
        sample_pairs = []
        with open(args.paired_list, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:
                    pair = {
                        'target': row[0],
                        'ref_same': row[1],
                        'ref_diff': row[2],
                        'label': row[3] if len(row) >= 4 else f'pair_{len(sample_pairs)}'
                    }
                    if os.path.exists(pair['target']) and os.path.exists(pair['ref_same']) and os.path.exists(pair['ref_diff']):
                        sample_pairs.append(pair)
                    else:
                        print(f"[Attribution] Warning: skipping pair with missing files: {pair['label']}")

        if not sample_pairs:
            print("[Attribution] No valid paired samples found. Exiting.")
            return

        baseline_computer = None
        if args.baseline_type != 'zero':
            first_model = list(models_dict.values())[0]
            baseline_computer = BaselineComputer(
                model=first_model,
                target_length=200 * 160 + 240,
                device=args.device
            )

        analyzer = ECAPAAttributionAnalyzer(
            models_dict=models_dict,
            C=args.C,
            device=args.device,
            musan_path=args.musan_path,
            baseline_computer=baseline_computer
        )

        subdir_name = "_vs_".join(model_identifiers)
        save_dir = os.path.join(args.save_path, f"paired_{subdir_name}")
        print(f"[Attribution] Results will be saved to: {save_dir}")

        analyzer.analyze_and_save_paired(
            sample_pairs, save_dir,
            baseline_type=args.baseline_type,
            objective=args.objective
        )
        print(f"[Attribution] Done! Results saved to {save_dir}")

    print("\n[Attribution] Selected audio files:")
    for i, s in enumerate(samples if args.mode == 'legacy' else [p['target'] for p in sample_pairs]):
        print(f"  {i+1}: {s}")

if __name__ == "__main__":
    main()