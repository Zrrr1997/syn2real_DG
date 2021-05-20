import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from utils.utils import AverageMeter, calc_topk_accuracy, ConfusionMeter, write_log, balanced_acc


def test_video_stream(data_loader, model, criterion, epoch, cuda_device, args):
    test_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                  "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                  "time_forward":       AverageMeter(locality=args.print_freq),
                  "time_backward":      AverageMeter(locality=args.print_freq),
                  "time_all":           AverageMeter(locality=args.print_freq),

                  "total_loss":         AverageMeter(locality=args.print_freq),

                  "accuracy":           {"top1": AverageMeter(locality=args.print_freq),
                                         "top3": AverageMeter(locality=args.print_freq),
                                         "top5": AverageMeter(locality=args.print_freq)}
                  }

    confusion_mat = ConfusionMeter(args.num_classes)

    model.eval()

    start_time = time.perf_counter()
    all_time = time.perf_counter()

    res_dict = {"vid_id": [], "chunk": [], "label": [], "pred1": [], "pred2": [], "pred3": [], "pred4": [], "pred5": []}
    res_scores = []
    with torch.no_grad():
        for idx, out in tqdm(enumerate(data_loader), total=len(data_loader)):
            vid_seqs, labels, vid_ids = out["vclip"], out["label"], out["vid_id"]
            res_dict["vid_id"].extend(vid_ids)
            res_dict["label"].extend(list(labels.numpy()))

            if "chunk" in out.keys():
                chunks = out["chunk"]
                res_dict["chunk"].extend(list(chunks.numpy()))

            batch_size = len(vid_seqs)

            test_stats["time_data_loading"].update(time.perf_counter() - start_time)
            start_time = time.perf_counter()

            vid_seqs = vid_seqs.to(cuda_device)
            if not args.test_prediction_only:
                labels = labels.to(cuda_device)

            test_stats["time_cuda_transfer"].update(time.perf_counter() - start_time)
            start_time = time.perf_counter()

            scores = model(vid_seqs)

            test_stats["time_forward"].update(time.perf_counter() - start_time)

            del vid_seqs

            if not args.test_prediction_only:
                loss = criterion(scores, labels)

            _, pred = torch.max(scores, dim=1)
            _, preds = torch.topk(scores, k=5, dim=1)

            for k in range(preds.shape[1]):
                res_dict[f"pred{k + 1}"].extend(list(preds[:, k].cpu().numpy()))

            if not args.test_prediction_only:
                confusion_mat.update(pred, labels.view(-1).byte())

                top1, top3, top5 = calc_topk_accuracy(scores, labels, (1, 3, 5))

                test_stats["total_loss"].update(loss.item(), batch_size)
                test_stats["accuracy"]["top1"].update(top1.item(), batch_size)
                test_stats["accuracy"]["top3"].update(top3.item(), batch_size)
                test_stats["accuracy"]["top5"].update(top5.item(), batch_size)

                del labels, loss

            res_scores.append(scores.cpu().numpy())

            del scores

            test_stats["time_all"].update(time.perf_counter() - all_time)

            start_time = time.perf_counter()
            all_time = time.perf_counter()

    res_df = pd.DataFrame(res_dict)
    
    csv_save_path = os.path.join(args.log_path, f'results_test_{args.dataset}_{args.split_policy}_'
                                              f'{args.model_vid}_{args.seq_len}f.csv')
    res_df.to_csv(csv_save_path, index=False)
    balanced_acc(csv_path = csv_save_path)

    all_scores = np.stack(res_scores)
    np.save(os.path.join(args.log_path, f'results_test_{args.dataset}_{args.split_policy}_'
                                        f'{args.model_vid}_{args.seq_len}f_scores.npy'), all_scores)

    print("Finished all predictions and saved them to file.")

    if not args.test_prediction_only:
        print("Saving log data and conf matrices...")
        result_str = 'Loss {loss:.4f}\t Acc top1: {top1:.4f} Acc top3: {top3:.4f} Acc top5: {top5:.4f} \t'.format(
            loss=test_stats["total_loss"].avg,
            top1=test_stats["accuracy"]["top1"].avg,
            top3=test_stats["accuracy"]["top3"].avg,
            top5=test_stats["accuracy"]["top5"].avg)

        print(result_str)

        write_log(content=result_str, epoch=epoch, filename=os.path.join(args.log_path,
                                                                         f'log_test_split_{args.dataset}.md'))

        confusion_mat.plot_mat(os.path.join(args.log_path, f'confm_test_split_{args.dataset}_an.svg'), annotate=True,
                               dictionary=data_loader.dataset.action_dict_decode)
        confusion_mat.plot_mat(os.path.join(args.log_path, f'confm_test_split_{args.dataset}.svg'), annotate=False,
                               dictionary=data_loader.dataset.action_dict_decode)
        print("Done.")

    return None
