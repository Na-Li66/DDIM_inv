# Table 3:
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "sr_averagepooling" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo

# Table 4:
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "inpainting" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "inpainting" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "inpainting" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "inpainting" --deg_scale 4.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo

# Table 5:
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "deblur_gauss" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "deblur_gauss" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "deblur_gauss" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "deblur_gauss" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo

# Table 6:
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "cs_walshhadamard" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config imagenet_256.yml --path_y imagenet --eta 0.0 16.0 --add_noise --deg "cs_walshhadamard" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "cs_walshhadamard" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 1 -i demo
python main.py --config celeba-hq.yml --path_y celeba/face --eta 0.0 16.0 --add_noise --deg "cs_walshhadamard" --deg_scale 5.0 --sigma_y 0.05 --NFE 100 --ntrials 4 -i demo
