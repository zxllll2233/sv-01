import argparse
import os
import gc
import torch
import random
import glob
import sys
import numpy as np
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
    parser.add_argument('--initial_model_2', type=str, default="/home/zhangxl24/SpeakerRecongnition/voiceprint/Baseline_noise_Spec/exp/vox1/model512/model_0089.model", help='Path to model 2')
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
                        choices=['legacy', 'paired', 'reliability'],
                        help='Attribution mode: legacy, paired, or reliability (deletion/insertion AUC test)')
    parser.add_argument('--paired_list', type=str, default=None,
                        help='Path to paired sample list CSV file (for paired/reliability mode). Format: target_path,ref_same_path,ref_diff_path,label')
    parser.add_argument('--baseline_type', type=str, default='zero',
                        choices=['zero', 'global_mean', 'speaker_mean', 'cross_speaker_mean'],
                        help='Baseline type for IG')
    parser.add_argument('--objective', type=str, default='cosine_sim',
                        choices=['cosine_sim', 'l2_norm'],
                        help='Attribution objective function')
    parser.add_argument('--add_noise', action='store_true',
                        help='Add noisy audio as right-side comparison block')
    parser.add_argument('--snr', type=float, default=None,
                        help='Fixed SNR (dB) for noise addition. If not set, random SNR from musan category ranges')
    # Reliability test parameters
    parser.add_argument('--del_ins_ratios', type=str, default='0.05,0.1,0.15,0.2,0.3,0.4,0.5',
                        help='Comma-separated deletion/insertion ratios for reliability test')
    parser.add_argument('--del_ins_mode', type=str, default='freq_time',
                        choices=['freq_time', 'freq', 'time'],
                        help='Deletion/insertion mode: freq_time (per cell), freq (per frequency bin), time (per time frame)')
    parser.add_argument('--n_random', type=int, default=10,
                        help='Number of random baseline runs for reliability test')
    parser.add_argument('--n_steps', type=int, default=50,
                        help='Number of IG integration steps')

    args = parser.parse_args()

    # 确保保存目录存在
    os.makedirs(args.save_path, exist_ok=True)
    
    # 1. 收集模型路径（跳过空路径）
    models_paths = [p for p in [args.initial_model_1, args.initial_model_2, args.initial_model_3] if p]
    if not models_paths:
        print("[Attribution] Error: No model paths provided.")
        sys.exit(1)

    # 2. 预处理样本（在模型循环外只做一次）
    samples = []
    sample_pairs = []

    if args.mode == 'legacy':
        samples = get_samples(args)
        if not samples:
            print("[Attribution] Failed to get samples. Exiting.")
            return
    elif args.mode in ('paired', 'reliability'):
        if not args.paired_list or not os.path.exists(args.paired_list):
            print(f"[Attribution] Error: --paired_list is required for {args.mode} mode")
            return
        import csv
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
                    target_full = os.path.join(args.eval_path, pair['target']) if args.eval_path else pair['target']
                    same_full = os.path.join(args.eval_path, pair['ref_same']) if args.eval_path else pair['ref_same']
                    diff_full = os.path.join(args.eval_path, pair['ref_diff']) if args.eval_path else pair['ref_diff']
                    if os.path.exists(target_full) and os.path.exists(same_full) and os.path.exists(diff_full):
                        sample_pairs.append(pair)
                    else:
                        print(f"[Attribution] Warning: skipping pair with missing files: {pair['label']}")
        if not sample_pairs:
            print("[Attribution] No valid paired samples found. Exiting.")
            return

    # 3. Paired mode: collect results across all models, visualize once
    if args.mode == 'paired':
        from attribution.analyzer import load_audio_as_tensor, compute_fbank, visualize_attribution_6row

        paper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'result')
        save_dir = os.path.join(paper_dir, "paired_all_models")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Attribution] Paired mode, saving to: {save_dir}")

        all_audio_paths = glob.glob(os.path.join(args.eval_path, "**/*.wav"), recursive=True)
        print(f"[Attribution] Found {len(all_audio_paths)} audio files for baseline computation")

        for i, pair in enumerate(sample_pairs):
            target_path = os.path.join(args.eval_path, pair['target']) if args.eval_path else pair['target']
            ref_same_path = os.path.join(args.eval_path, pair['ref_same']) if args.eval_path else pair['ref_same']
            ref_diff_path = os.path.join(args.eval_path, pair['ref_diff']) if args.eval_path else pair['ref_diff']
            label = pair.get('label', f'sample_{i}')

            print(f"\n[Paired Attribution] Analyzing {label}...")

            target_wav, target_tensor = load_audio_as_tensor(target_path, args.device)
            _, ref_same_tensor = load_audio_as_tensor(ref_same_path, args.device)
            _, ref_diff_tensor = load_audio_as_tensor(ref_diff_path, args.device)

            noisy_tensor = None
            noise_wav = None
            noise_info = ""
            if args.add_noise:
                noise_files = glob.glob(os.path.join(args.musan_path, '*/*/*.wav')) if args.musan_path else []
                if noise_files:
                    noise_path = random.choice(noise_files)
                    noise_wav, _ = load_audio_as_tensor(noise_path, args.device)
                    clean_db = 10 * np.log10(np.mean(target_wav ** 2) + 1e-4)
                    noise_db = 10 * np.log10(np.mean(noise_wav ** 2) + 1e-4)
                    snr_val = args.snr if args.snr is not None else random.uniform(0, 15)
                    scale = np.sqrt(10 ** ((clean_db - noise_db - snr_val) / 10))
                    noisy_wav = target_wav + scale * noise_wav
                    noisy_tensor = torch.FloatTensor(noisy_wav).unsqueeze(0).to(args.device)
                    noise_info = f"{os.path.basename(noise_path)} (SNR={snr_val:.1f}dB)"

            all_l2_attrs = {}
            all_l2_noisy_attrs = {}
            all_cos_clean_attrs = {}
            all_cos_noisy_attrs = {}
            fbank_clean = None
            fbank_noise = None
            fbank_noisy = None

            for mi, model_path in enumerate(models_paths):
                if not os.path.exists(model_path):
                    print(f"Error: Model file {model_path} not found. Skipping.")
                    continue

                try:
                    model = ECAPAModel(**vars(args))
                    model.load_parameters(model_path)
                    model.eval()
                except Exception as e:
                    print(f"Error loading model {model_path}: {e}. Skipping.")
                    continue

                try:
                    path_parts = os.path.normpath(model_path).split(os.sep)
                    epoch_str = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]
                    dir_name = path_parts[-5] if len(path_parts) >= 5 else (path_parts[-2] if len(path_parts) >= 2 else f"Model_{mi+1}")
                    model_name = f"{dir_name}_{epoch_str}"
                except Exception:
                    model_name = f"Model_{mi+1}"

                print(f"  [Model {mi+1}/{len(models_paths)}] {model_name}")

                if fbank_clean is None:
                    fbank_clean = compute_fbank(model.speaker_encoder, target_tensor)
                    fbank_noisy = compute_fbank(model.speaker_encoder, noisy_tensor) if noisy_tensor is not None else fbank_clean
                    if noise_wav is not None:
                        noise_tensor = torch.FloatTensor(noise_wav).unsqueeze(0).to(args.device)
                        fbank_noise = compute_fbank(model.speaker_encoder, noise_tensor)
                    else:
                        fbank_noise = fbank_noisy

                baseline_computer = None
                if args.baseline_type != 'zero':
                    baseline_computer = BaselineComputer(
                        model=model.speaker_encoder,
                        target_length=200 * 160 + 240,
                        device=args.device
                    )

                speaker_audio_paths = None
                exclude_speaker_paths = None
                if args.baseline_type != 'zero' and all_audio_paths:
                    target_dir = os.path.dirname(target_path)
                    speaker_audio_paths = [p for p in all_audio_paths if os.path.dirname(p) == target_dir]
                    if args.baseline_type == 'cross_speaker_mean':
                        exclude_speaker_paths = speaker_audio_paths

                models_dict = {model_name: model.speaker_encoder}
                analyzer = ECAPAAttributionAnalyzer(
                    models_dict=models_dict,
                    C=args.C,
                    device=args.device,
                    musan_path=args.musan_path,
                    baseline_computer=baseline_computer
                )

                attrs = analyzer.compute_model_attribution(
                    target_tensor, ref_same_tensor, ref_diff_tensor,
                    noisy_tensor=noisy_tensor,
                    baseline_type=args.baseline_type,
                    speaker_audio_paths=speaker_audio_paths,
                    all_audio_paths=all_audio_paths,
                    exclude_speaker_paths=exclude_speaker_paths
                )

                all_l2_attrs[model_name] = attrs[model_name]['l2_norm']
                all_l2_noisy_attrs[model_name] = attrs[model_name].get('l2_norm_noisy', attrs[model_name]['l2_norm'])
                all_cos_clean_attrs[model_name] = attrs[model_name]['cosine_sim_diff']
                all_cos_noisy_attrs[model_name] = attrs[model_name].get('cosine_sim_noisy_diff', attrs[model_name]['cosine_sim_diff'])

                del model, models_dict, analyzer, baseline_computer
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                print(f"  [Model {mi+1}] done, GPU freed.")

            if all_l2_attrs:
                model_names = list(all_l2_attrs.keys())
                save_path = os.path.join(save_dir, f"{label}_attribution.png")
                visualize_attribution_6row(
                    save_path, model_names,
                    fbank_clean, fbank_noise, fbank_noisy,
                    all_l2_attrs, all_l2_noisy_attrs, all_cos_clean_attrs, all_cos_noisy_attrs,
                    audio_label=os.path.basename(target_path),
                    noise_info=noise_info,
                    baseline_type=args.baseline_type
                )
                print(f"[Paired Attribution] Saved: {save_path}")

        print(f"\n[Attribution] All models completed.")
        return

    # 3.5. Reliability mode: Deletion/Insertion AUC test
    if args.mode == 'reliability':
        from attribution.reliability import (
            batch_deletion_insertion_test,
            plot_reliability_curves,
            plot_multi_model_comparison,
            plot_method_comparison,
        )

        paper_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'paper', 'result')
        save_dir = os.path.join(paper_dir, "reliability_test")
        os.makedirs(save_dir, exist_ok=True)
        print(f"[Reliability] Saving to: {save_dir}")

        ratios = [float(r) for r in args.del_ins_ratios.split(',')]
        del_ins_mode = args.del_ins_mode

        all_cosine_results = {}
        all_l2_results = {}

        for mi, model_path in enumerate(models_paths):
            if not os.path.exists(model_path):
                print(f"Error: Model file {model_path} not found. Skipping.")
                continue

            try:
                model = ECAPAModel(**vars(args))
                model.load_parameters(model_path)
                model.eval()
            except Exception as e:
                print(f"Error loading model {model_path}: {e}. Skipping.")
                continue

            try:
                path_parts = os.path.normpath(model_path).split(os.sep)
                epoch_str = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]
                dir_name = path_parts[-5] if len(path_parts) >= 5 else (path_parts[-2] if len(path_parts) >= 2 else f"Model_{mi+1}")
                model_name = f"{dir_name}_{epoch_str}"
            except Exception:
                model_name = f"Model_{mi+1}"

            print(f"\n[Reliability] Model {mi+1}/{len(models_paths)}: {model_name}")

            print(f"  Running cosine_sim_diff attribution...")
            cos_result = batch_deletion_insertion_test(
                model=model.speaker_encoder,
                sample_pairs=sample_pairs,
                attribution_method='cosine_sim_diff',
                ratios=ratios,
                mode=del_ins_mode,
                n_random=args.n_random,
                n_steps=args.n_steps,
                device=args.device,
                eval_path=args.eval_path,
            )
            all_cosine_results[model_name] = cos_result

            print(f"  Running l2_norm attribution...")
            l2_result = batch_deletion_insertion_test(
                model=model.speaker_encoder,
                sample_pairs=sample_pairs,
                attribution_method='l2_norm',
                ratios=ratios,
                mode=del_ins_mode,
                n_random=args.n_random,
                n_steps=args.n_steps,
                device=args.device,
                eval_path=args.eval_path,
            )
            all_l2_results[model_name] = l2_result

            plot_reliability_curves(
                cos_result,
                save_path=os.path.join(save_dir, f"{model_name}_cosine_sim_reliability.png"),
                model_name=model_name,
                attribution_method='cosine_sim_diff',
                mode=del_ins_mode,
            )
            plot_reliability_curves(
                l2_result,
                save_path=os.path.join(save_dir, f"{model_name}_l2_norm_reliability.png"),
                model_name=model_name,
                attribution_method='l2_norm',
                mode=del_ins_mode,
            )
            plot_method_comparison(
                cos_result, l2_result,
                save_path=os.path.join(save_dir, f"{model_name}_method_comparison.png"),
                model_name=model_name,
                mode=del_ins_mode,
            )

            print(f"  Deletion AUC:   cos_sim={cos_result['deletion_auc']:.4f}  l2={l2_result['deletion_auc']:.4f}")
            print(f"  Insertion AUC:  cos_sim={cos_result['insertion_auc']:.4f}  l2={l2_result['insertion_auc']:.4f}")
            print(f"  Random Del AUC: cos_sim={cos_result['random_deletion_auc']:.4f}  l2={l2_result['random_deletion_auc']:.4f}")

            del model
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
            print(f"  [Model {mi+1}] done, GPU freed.")

        if len(all_cosine_results) > 1:
            plot_multi_model_comparison(
                all_cosine_results,
                save_path=os.path.join(save_dir, "multi_model_cosine_sim.png"),
                attribution_method='cosine_sim_diff',
                mode=del_ins_mode,
            )
            plot_multi_model_comparison(
                all_l2_results,
                save_path=os.path.join(save_dir, "multi_model_l2_norm.png"),
                attribution_method='l2_norm',
                mode=del_ins_mode,
            )

        print(f"\n[Reliability] All models completed. Results saved to: {save_dir}")
        return

    # 4. Legacy mode: per-model loop
    for i, model_path in enumerate(models_paths):
        print(f"\n{'='*60}")
        print(f"[Attribution] Model {i+1}/{len(models_paths)}: {model_path}")
        print(f"{'='*60}")

        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found. Skipping.")
            continue

        try:
            model = ECAPAModel(**vars(args))
            model.load_parameters(model_path)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_path}: {e}. Skipping.")
            continue

        try:
            path_parts = os.path.normpath(model_path).split(os.sep)
            epoch_str = os.path.splitext(os.path.basename(model_path))[0].split('_')[-1]
            dir_name = path_parts[-5] if len(path_parts) >= 5 else (path_parts[-2] if len(path_parts) >= 2 else f"Model_{i+1}")
            model_name = f"{dir_name}_{epoch_str}"
        except Exception:
            model_name = f"Model_{i+1}"

        models_dict = {model_name: model.speaker_encoder}
        save_dir = os.path.join(args.save_path, model_name)

        if args.mode == 'legacy':
            print(f"[Attribution] Legacy mode, saving to: {save_dir}")

            analyzer = ECAPAAttributionAnalyzer(
                models_dict=models_dict,
                C=args.C,
                device=args.device,
                musan_path=args.musan_path
            )
            analyzer.analyze_and_save(samples, save_dir)

            print(f"[Attribution] Model {model_name} done! Saved to {save_dir}")
            del model, models_dict, analyzer
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except RuntimeError:
                pass
        print(f"[Attribution] GPU memory freed.")

    print(f"\n[Attribution] All {len(models_paths)} models completed.")

if __name__ == "__main__":
    main()