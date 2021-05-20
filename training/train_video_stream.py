import time

import torch

from utils.utils import write_out_checkpoint, AverageMeter, calc_topk_accuracy, write_out_images


def training_loop_video_stream(model, optimizer, lr_schedule, criterion, dl_train, dl_val, writer_train, writer_val,
                               args,
                               cuda_device):
    iteration = args.start_iteration
    best_val_acc = args.best_val_acc
    best_val_loss = args.best_val_loss
    best_train_acc = args.best_train_acc
    best_train_loss = args.best_train_loss

    # Main loop
    for epoch in range(args.start_epoch, args.epochs):
        # Single epoch training
        iteration, train_loss, train_acc = train_video_stream(dl_train, model, optimizer, criterion,
                                                              epoch, iteration, args, writer_train, cuda_device)

        # Single epoch validation
        val_loss, val_acc = validate_video_stream(dl_val, model, criterion, cuda_device,
                                                  epoch, args, writer_val) if not args.skip_val else (0, 0)

        # This is to decide if models have to be written to file.
        best_val_acc = val_acc if best_val_acc is None or val_acc > best_val_acc else best_val_acc
        best_val_loss = val_loss if best_val_loss is None or val_loss < best_val_loss else best_val_loss

        best_train_acc = train_acc if best_train_acc is None or train_acc > best_train_acc else best_train_acc
        best_train_loss = train_loss if best_train_loss is None or train_loss < best_train_loss else best_train_loss

        write_out_checkpoint(epoch, iteration, model, optimizer, args,
                             train_loss, train_acc, val_loss, val_acc,
                             best_train_loss, best_train_acc, best_val_loss, best_val_acc)

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))


def train_video_stream(dl_train, model, optimizer, criterion,
                       epoch, iteration, args, writer_train, cuda_device):
    tr_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                "time_forward":       AverageMeter(locality=args.print_freq),
                "time_scoring":       AverageMeter(locality=args.print_freq),
                "time_backward":      AverageMeter(locality=args.print_freq),
                "time_all":           AverageMeter(locality=args.print_freq),

                "total_loss":         AverageMeter(locality=args.print_freq),

                "accuracy":           {"top1": AverageMeter(locality=args.print_freq),
                                       "top3": AverageMeter(locality=args.print_freq),
                                       "top5": AverageMeter(locality=args.print_freq)}
                }

    model.train()

    dl_time = time.perf_counter()
    all_time = time.perf_counter()

    for idx, out in enumerate(dl_train):
        vid_seqs, labels = out["vclip"], out["label"]
        batch_size = len(vid_seqs)

        tr_stats["time_data_loading"].update(time.perf_counter() - dl_time)

        # Visualize images for tensorboard.
        if iteration == 0 or iteration == args.print_freq:
            write_out_images(vid_seqs, writer_train, iteration)

        # Cuda Transfer
        s_cud_time = time.perf_counter()

        vid_seqs = vid_seqs.to(cuda_device)
        labels = labels.to(cuda_device)

        e_cud_time = time.perf_counter()
        tr_stats["time_cuda_transfer"].update(e_cud_time - s_cud_time)

        # Forward pass: Calculation
        s_forw_time = time.perf_counter()

        scores = model(vid_seqs)

        e_forw_time = time.perf_counter()
        tr_stats["time_forward"].update(e_forw_time - s_forw_time)

        # Calculate Accuracies
        top1, top3, top5 = calc_topk_accuracy(scores, labels, (1, min(3, batch_size), min(5, batch_size)))

        tr_stats["accuracy"]["top1"].update(top1.item(), batch_size)
        tr_stats["accuracy"]["top3"].update(top3.item(), batch_size)
        tr_stats["accuracy"]["top5"].update(top5.item(), batch_size)

        # Loss Calculation and backward pass.
        s_back_time = time.perf_counter()

        loss = criterion(scores, labels)

        tr_stats["total_loss"].update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        e_back_time = time.perf_counter()
        tr_stats["time_backward"].update(e_back_time - s_back_time)

        tr_stats["time_all"].update(time.perf_counter() - all_time)

        if idx % args.print_freq == 0:
            # Write stats to tensorboard and print stats to console
            write_stats_batch_contrast_iteration(tr_stats, writer_train, iteration)
            print_tr_stats_loc_avg(tr_stats, epoch, idx, len(dl_train), tr_stats["time_all"].local_avg)

        del vid_seqs, labels, scores, loss

        iteration += 1

        dl_time = time.perf_counter()
        all_time = time.perf_counter()
        # Next iteration

    # Write stats to tensorboard and print average timings to console (for whole epoch)
    write_stats_batch_contrast_epoch(tr_stats, writer_train, epoch)
    print_tr_stats_timings_avg(tr_stats)

    return iteration, tr_stats["total_loss"].avg, tr_stats["accuracy"]["top1"].avg


def validate_video_stream(data_loader, model, criterion, cuda_device, epoch, args, writer_val):
    val_stats = {"time_data_loading":  AverageMeter(locality=args.print_freq),
                 "time_cuda_transfer": AverageMeter(locality=args.print_freq),
                 "time_forward":       AverageMeter(locality=args.print_freq),
                 "time_backward":      AverageMeter(locality=args.print_freq),
                 "time_all":           AverageMeter(locality=args.print_freq),

                 "total_loss":         AverageMeter(locality=args.print_freq),

                 "accuracy":           {"top1": AverageMeter(locality=args.print_freq),
                                        "top3": AverageMeter(locality=args.print_freq),
                                        "top5": AverageMeter(locality=args.print_freq)}
                 }

    model.eval()

    start_time = time.perf_counter()
    all_time = time.perf_counter()

    with torch.no_grad():
        for idx, out in enumerate(data_loader):
            vid_seqs, labels = out["vclip"], out["label"]
            batch_size = len(vid_seqs)

            val_stats["time_data_loading"].update(time.perf_counter() - start_time)
            start_time = time.perf_counter()

            vid_seqs = vid_seqs.to(cuda_device)
            labels = labels.to(cuda_device)

            val_stats["time_cuda_transfer"].update(time.perf_counter() - start_time)
            start_time = time.perf_counter()

            scores = model(vid_seqs)

            val_stats["time_forward"].update(time.perf_counter() - start_time)

            del vid_seqs

            loss = criterion(scores, labels)

            top1, top3, top5 = calc_topk_accuracy(scores, labels, (1, 3, 5))

            val_stats["total_loss"].update(loss.item(), batch_size)
            val_stats["accuracy"]["top1"].update(top1.item(), batch_size)
            val_stats["accuracy"]["top3"].update(top3.item(), batch_size)
            val_stats["accuracy"]["top5"].update(top5.item(), batch_size)

            del scores, labels, loss

            val_stats["time_all"].update(time.perf_counter() - all_time)

            start_time = time.perf_counter()
            all_time = time.perf_counter()

    print_val_avg(val_stats, epoch, args)

    write_val_stats_avg(val_stats, writer_val, epoch)

    return val_stats["total_loss"].avg, val_stats["accuracy"]["top1"].avg


def write_stats_batch_contrast_iteration(tr_stats, writer_train, iteration):
    writer_train.add_scalars('it/Train_Loss', {'loss': tr_stats["total_loss"].local_avg}, iteration)

    writer_train.add_scalars('it/Train_Accuracy', {'top1': tr_stats["accuracy"]["top1"].local_avg,
                                                   'top3': tr_stats["accuracy"]["top3"].local_avg,
                                                   'top5': tr_stats["accuracy"]["top5"].local_avg},
                             iteration)

    all_calced_timings = sum([tr_stats[tms].local_avg for tms in ["time_data_loading",
                                                                  "time_cuda_transfer",
                                                                  "time_forward",
                                                                  "time_scoring",
                                                                  "time_backward",
                                                                  ]])
    timing_dict = {'Loading Data':                 tr_stats["time_data_loading"].local_avg,
                   'Cuda Transfer':                tr_stats["time_cuda_transfer"].local_avg,
                   'Forward Pass':                 tr_stats["time_forward"].local_avg,
                   'Scoring':                      tr_stats["time_scoring"].local_avg,
                   'Backward Pass':                tr_stats["time_backward"].local_avg,
                   'Loading + Transfer + '
                   'Forward + Scoring + Backward': all_calced_timings,
                   'All':                          tr_stats["time_all"].local_avg
                   }

    writer_train.add_scalars('it/Batch-Wise_Timings', timing_dict, iteration)


def print_tr_stats_timings_avg(tr_stats):
    print('Batch-wise Timings:\n'
          f'Data Loading: {tr_stats["time_data_loading"].avg:.4f}s | '
          f'Cuda Transfer: {tr_stats["time_cuda_transfer"].avg:.4f}s | '
          f'Forward: {tr_stats["time_forward"].avg:.4f}s | '
          f'Backward: {tr_stats["time_backward"].avg:.4f}s | '
          f'All: {tr_stats["time_all"].avg:.4f}s\n')


def print_tr_stats_loc_avg(stats: dict, epoch, idx, batch_count, duration):
    print(f'Epoch: [{epoch}][{idx}/{batch_count}]\t '
          f'Loss {stats["total_loss"].local_avg:.6f} '
          '\tAcc: '
          f'top1 {stats["accuracy"]["top1"].local_avg:.4f} | '
          f'top3 {stats["accuracy"]["top3"].local_avg:.4f} | '
          f'top5 {stats["accuracy"]["top5"].local_avg:.4f} | '
          f'T:{duration:.2f}')


def write_stats_batch_contrast_epoch(tr_stats, writer_train, epoch):
    writer_train.add_scalars('ep/Losses', {'Training Loss': tr_stats["total_loss"].avg}, epoch)

    writer_train.add_scalars('ep/Accuracies', {'Train Acc': tr_stats["accuracy"]["top1"].avg}, epoch)

    writer_train.add_scalars('ep/Train_Accuracy', {'top1': tr_stats["accuracy"]["top1"].avg,
                                                   'top3': tr_stats["accuracy"]["top3"].avg,
                                                   'top5': tr_stats["accuracy"]["top5"].avg}, epoch)


def print_val_avg(val_stats, epoch, args):
    print(f'[{epoch}/{args.epochs}] Loss {val_stats["total_loss"].avg:.4f}\t'
          f'Acc: '
          f'top1 {val_stats["accuracy"]["top1"].avg:.4f}; '
          f'top3 {val_stats["accuracy"]["top3"].avg:.4f}; '
          f'top5 {val_stats["accuracy"]["top5"].avg:.4f}\n')


def write_val_stats_avg(val_stats, writer_val, epoch):
    writer_val.add_scalars('ep/Losses', {'Validation Loss': val_stats["total_loss"].avg}, epoch)

    writer_val.add_scalars('ep/Accuracies', {'Val Acc': val_stats["accuracy"]["top1"].avg}, epoch)

    writer_val.add_scalars('ep/Val_Accuracy', {"top1": val_stats["accuracy"]["top1"].avg,
                                               "top3": val_stats["accuracy"]["top3"].avg,
                                               "top5": val_stats["accuracy"]["top5"].avg}, epoch)

    writer_val.add_scalars('ep/Validation_Timings', {"Data Loading":  val_stats["time_data_loading"].avg,
                                                     "Cuda Transfer": val_stats["time_cuda_transfer"].avg,
                                                     "Forward Pass":  val_stats["time_forward"].avg,
                                                     "Total":         val_stats["time_all"].avg}, epoch)
