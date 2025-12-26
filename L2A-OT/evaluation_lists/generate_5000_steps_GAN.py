with open('evaluate_rgb_GAN', 'w') as f:
	for i in range (1, 10):
		f.write("main_SIMS_S3D.py --exp_tag rgb_GAN_" + str(5000 * i) + " --pretrained_model_DGC checkpoints/rgb_GAN/DGC_iteration_" + str(5000*i) +".pth --pretrained_model_G checkpoints/rgb_GAN/G_iteration_"+str(5000*i)+".pth --gpu 3 3  --test_classifier_only --modality_indices 3 --logs ./logs/EVAL/\n")
