with open('evaluate_l_of', 'w') as f:
	for i in range (1, 10):
		f.write("main_SIMS_S3D.py --exp_tag l_of_" + str(5000 * i) + " --pretrained_model_DGC checkpoints/limbs_optical_flow/DGC_iteration_" + str(5000*i) +".pth --pretrained_model_G checkpoints/second_GAN/GAN_training_second_paper_lambdasG_iteration_27000.pth --gpu 3 3  --test_classifier_only --modality_indices 1 2 --logs ./logs/EVAL/\n")
