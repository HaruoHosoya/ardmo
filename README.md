# A Cognitive Model for Learning Abstract Relational Structures from Memory-based Decision-Making Tasks

This is the official reference pytorch implementation of ARDMO (Abstract Relational Decision-making MOdel) by Haruo Hosoya [1].

# Dependencies

matplotlib==3.5.2

numpy==1.23.1

scikit_learn==1.1.3

scipy==1.9.1

torch==1.12.1

torchvision==0.13.1

tqdm==4.64.1

# Dataset generation

First, to train models for 1D or 2D task, you need to generate datasets.  For this, type:

 python prep_data1d.py
 
 python prep_data2d.py
 
This will create dataset files in a new folder datasets in the project folder.

# Model training and test for 1D task

For model training for 1D task, you can use the following command:

 python main1d.py --dataset [[dataset name]] --model_id [[model name]] --gpu [[gpu#]] --epochs [[#epochs]] --mode train

For example, for a model training just like in the paper, type:

 python main1d.py --dataset ds1d_cifar_1000_7_700 --model_id model_1d_1 --gpu 0 --epochs 8000 --mode train

This will create a new folder results/model_1d_1 and emit various intermediate result files therein.

To analyze the resulting model, type:

 python main1d.py --dataset ds1d_cifar_1000_7_700 --model_id model_1d_1 --gpu 0 --mode eval-full
 python main1d.py --dataset ds1d_cifar_1000_7_700 --model_id model_1d_1 --gpu 0 --mode show

This will emit analysis result files in the folder results/model_id_1.  
For example, fig_performance_adj.pdf shows the trace of relational inference performance; 
fig_performance_nonadj.pdf shows the trace of transitive inference score.

See other options by:

 python main1d.py --help

# Model training and test for 2D task

Similarly, for 2D task, use the following command:

  python main2d.py --dataset [[dataset name]] --model_id [[model name]] --gpu [[gpu#]] --epochs [[#epochs]] --mode train

For training just like in the paper, type:

  python main2d.py --dataset ds2d_cifar_1000_4_700 --model_id model_2d_1 --gpu 0 --epochs 6000 --mode train

For analysis, type:

 python main2d.py --dataset ds2d_cifar_1000_4_700 --model_id model_2d_1 --gpu 0 --mode eval-full
 
 python main2d.py --dataset ds2d_cifar_1000_4_700 --model_id model_2d_1 --gpu 0 --mode show

See other options by:

 python main2d.py --help

# References and contact
If you publish a paper based on this code/data, please cite [1].  If you have difficulty or question, please contact with the author.

[1] Haruo Hosoya.  A Cognitive Model for Learning Abstract Relational Structures from Memory-based Decision-Making Tasks.   ICLR 2024.  https://openreview.net/forum?id=KC58bVmxyN


