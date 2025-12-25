import os
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from testing.generator import Generator

from utils.utils import AverageMeter, calc_topk_accuracy, ConfusionMeter, write_log, balanced_acc


def test_video_stream(data_loader, model, criterion, epoch, cuda_device, args):
    G = Generator(c_dim = 2 * 4).to(cuda_device) # c_dim = K_n + K_s = 2 * K_s

    for name, param in G.named_parameters():
        param.requires_grad = False
    G.load_state_dict(torch.load(args.G_path, map_location='cuda:0'))


    g_optimizer = torch.optim.Adam(G.parameters(), 0.001, (0.5, 0.999))
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
    res_embeddings = []
    res_labels = []
    with torch.no_grad():
        for idx, out in tqdm(enumerate(data_loader), total=len(data_loader)):
            if args.model_vid == 'YOLO_mlp':
                vid_seqs, labels, vid_ids  = out["detection"], out["label"], out["vid_id"]
                res_dict["vid_id"].extend(vid_ids)
                res_dict["label"].extend(list(labels.numpy()))
            elif args.model_vid == 's3d_yolo_fusion':
                vid_seqs, dets, labels, vid_ids = out["vclip"], out["detection"], out["label"], out["vid_id"]
                dets = dets.to(cuda_device)
                res_dict["vid_id"].extend(vid_ids)
                res_dict["label"].extend(list(labels.numpy()))
            else:
                vid_seqs, labels, vid_ids  = out["vclip"], out["label"], out["vid_id"]
                res_dict["vid_id"].extend(vid_ids)
                res_dict["label"].extend(list(labels.numpy()))


            vid_seqs = transform_to_novel_batch(vid_seqs, G, args, cuda_device)
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

            embeddings = model.module.embed(vid_seqs)
            if args.model_vid == 's3d_yolo_fusion':  
                scores = model(vid_seqs, dets)
            else:
                scores = model(vid_seqs)
            res_labels.append(labels.cpu().numpy())

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

            res_embeddings.append(embeddings.cpu().numpy())

            del scores
            del embeddings

            test_stats["time_all"].update(time.perf_counter() - all_time)

            start_time = time.perf_counter()
            all_time = time.perf_counter()

    res_df = pd.DataFrame(res_dict)
    
    csv_save_path = os.path.join(args.log_path, f'results_test_{args.dataset}_{args.split_policy}_'
                                              f'{args.model_vid}_{args.seq_len}f.csv')
    res_df.to_csv(csv_save_path, index=False)
    balanced_acc(csv_path = csv_save_path)

    all_scores = np.stack(res_scores)
    all_embeddings = np.stack(res_embeddings)
    all_labels = np.stack(res_labels)

    np.save(os.path.join(args.log_path, f'results_test_{args.dataset}_{args.split_policy}_'
                                        f'{args.model_vid}_{args.seq_len}f_scores.npy'), all_scores)
    np.save(os.path.join(args.log_path, f'results_test_{args.dataset}_{args.split_policy}_'
                                        f'{args.model_vid}_{args.seq_len}f_embeddings.npy'), all_embeddings)
    np.save(os.path.join(args.log_path, f'results_test_{args.dataset}_{args.split_policy}_'
                                        f'{args.model_vid}_{args.seq_len}f_labels.npy'), all_labels)
    print("Saved", len(res_embeddings), "embeddings")
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
def transform_to_novel_batch(x_all_batch, G, args, generator_device):
        G.train() 

        modality_indices = []
        modalities = ['heatmaps', 'limbs', 'optical_flow', 'rgb']
        num_domains = len(modalities)
        num_aug_domains = num_domains
        data = []
        curr = 0

        with torch.no_grad():
            x_all_batch = x_all_batch.transpose(1, 2) # switch Sequence and Channel dimension to combine sequence and batch dim

            x_all_batch = x_all_batch.contiguous().view(-1, *x_all_batch.shape[2:]).transpose(0, 1) # (batch_size, 8, 16, 112, 122) -> (8, 16*batch_size, 112, 112)
            for i, c in enumerate(args.n_channels_each_modality):
                mod = x_all_batch[curr:curr+c].transpose(0, 1)
                if c == 1:
                    mod = torch.cat((mod, mod, mod), 1)
                curr+=c
                data.append(mod)
            for i in range(4):
                if modalities[i] in args.modalities:
                    modality_indices.append(i)



        
            for i, index in enumerate(modality_indices): # go over all source domains

                x_real = data[i].to(generator_device) # Extract source modality for the input

                label_org = torch.zeros(x_real.size(0), num_domains + num_aug_domains) # Label of source domain
                label_org[:,index] = 1.0
                label_org = label_org.to(generator_device)
                new_idx = num_domains + index

                label_trg = torch.zeros(x_real.size(0), num_domains + num_aug_domains)# label of target domain (shifted with just K_s)
                label_trg[:,new_idx] = 1.0
                label_trg = label_trg.to(generator_device)

                # Original-to-target domain.

                x_fake = G(x_real, label_trg).to(generator_device)


                # Store all fake images
                if index == 0 or (modality_indices[0] == 1 and index == 1):
                    full_fakes = x_fake
                    #fakes = x_fake[:,0,:,:].unsqueeze(1)
                    fakes = torch.mean(x_fake, dim=1, keepdim=True)
                elif index == 1:
                    full_fakes = torch.cat((full_fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                    #fakes = torch.cat((fakes, x_fake[:,0,:,:].unsqueeze(1)), 1) # Concatenate generated domains along channel dimension
                    fakes = torch.cat((fakes, torch.mean(x_fake, dim=1, keepdim=True)), 1) # Concatenate generated domains along channel dimension
                else:
                    if index == args.modality_indices[0]:
                        full_fakes = x_fake
                        fakes = x_fake
                    else:

                        full_fakes = torch.cat((full_fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                        fakes = torch.cat((fakes, x_fake), 1) # Concatenate generated domains along channel dimension
                del x_fake, x_real, label_org, label_trg

            fakes = fakes.view(args.batch_size, args.seq_len, args.n_channels, args.img_dim, args.img_dim).transpose(1, 2) # bring back to original dimensions and swap channels and sequence
            #x_all_batch = x_all_batch.transpose(0, 1) # (C, B*S, W, H) -> (B*S, C, W, H)
            #x_all_batch = x_all_batch.view(args.batch_size, n_channels, 16, 112, 112)
            assert fakes.shape == (args.batch_size, args.n_channels, args.seq_len,  args.img_dim,  args.img_dim)
        return fakes


