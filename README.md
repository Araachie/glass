<h1 align="center">
  <br>
	[ECCV 2022] Controllable Video Generation through Global and Local Motion Dynamics
  <br>
</h1>
  <p align="center">
    <a href="https://araachie.github.io">Aram Davtyan</a> •
    <a href="https://www.cvg.unibe.ch/people/favaro">Paolo Favaro</a>
  </p>
<h4 align="center">Official repository of the paper</h4>

<h4 align="center">ECCV 2022</h4>

<h4 align="center"><a href="https://araachie.github.io/glass/">Project Website</a> • <a href="https://arxiv.org/abs/2204.06558">Arxiv</a>

#
> **Abstract:** *We present GLASS, a method for Global and Local Action-driven Sequence Synthesis.
> GLASS is a generative model that is trained on video sequences in an unsupervised manner and 
> that can animate an input image at test time. The method learns to segment frames into foreground-background
> layers and to generate transitions of the foregrounds over time through a global and local action representation. 
> Global actions are explicitly related to 2D shifts, while local actions are instead related to 
> (both geometric and photometric) local deformations. GLASS uses a recurrent neural network to transition 
> between frames and is trained through a reconstruction loss. We also introduce W-Sprites (Walking Sprites), 
> a novel synthetic dataset with a predefined action space. We evaluate our method on both W-Sprites and 
> real datasets, and find that GLASS is able to generate realistic video sequences from a single input image 
> and to successfully learn a more advanced action space than in prior work.*

## Citation

The paper is to appear in the Proceedings of the 17th European Conference on Computer Vision in 2022. 
In the meantime we suggest using the arxiv preprint bibref.

Davtyan, A. & Favaro, P. (2022). Controllable Video Generation through Global and Local Motion Dynamics.
arXiv preprint arXiv:2204.06558.

    @misc{https://doi.org/10.48550/arxiv.2204.06558,
      doi = {10.48550/ARXIV.2204.06558},
      url = {https://arxiv.org/abs/2204.06558},
      author = {Davtyan, Aram and Favaro, Paolo},
      keywords = {Computer Vision and Pattern Recognition (cs.CV), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
      title = {Controllable Video Generation through Global and Local Motion Dynamics},
      publisher = {arXiv},
      year = {2022},
      copyright = {Creative Commons Attribution 4.0 International}
    }


## Prerequisites

For convenience, we provide an `env.yml` file that can be used to install the required packages to a `conda` environment 
with the following command 

```conda env create -f evn.yml```

## Datasets

### Tennis and BAIR

Collect the datasets following instruction from the [official PVG repository](https://github.com/willi-menapace/PlayableVideoGeneration#preparing-datasets).
Put them into the `./data` folder, if you want to use the default configs, or specify the path to the dataset in the corresponding config otherwise.
For instance, by default the code expects to find the *Tennis* dataset in the `./data/tennis` folder that in its turn contains 3 subfolders for *train*, *val* and *test* splits of the dataset. 

### W-Sprites

To generate *W-Sprites* dataset, first activate the environment specified in the `data/wsprites/env.yml` file and then run the following commands:

```angular2html
1. cd data/wsprites/sprites
2. python random_character.py
3. python frame_to_npy.py
4. cd ../../../
5. python -m data.wsprites.generate
```

This will create training sequences in the `data/wsprites/data` folder.

### Custom datasets

Use the `convert_video_directory.py` script from the [official PVG repository](https://github.com/willi-menapace/PlayableVideoGeneration#custom-datasets) to convert a custom video dataset to the appropriate format.

## Training

To reproduce results from the paper, use the `train.py` script from the repository.
Usage example:

```
python train.py --config configs/wsprites.yaml --run-name test-run --wandb
```

The output of `python train.py --help` is as follows

```
> python train.py --help
usage: train.py [-h] --run-name RUN_NAME --config CONFIG [--resume-step RESUME_STEP] [--num-gpus NUM_GPUS] [--random-seed RANDOM_SEED] [--wandb]

optional arguments:
  -h, --help            show this help message and exit
  --run-name RUN_NAME   Name of the current run.
  --config CONFIG       Path to the config file.
  --resume-step RESUME_STEP
                        Step to resume the training from.
  --num-gpus NUM_GPUS   Number of gpus to use for training.
                        By default uses all available gpus.
  --random-seed RANDOM_SEED
                        Random seed.
  --wandb               If defined, use wandb for logging.
```

Please, check example configs in the `configs` folder.

## Evaluation

The evaluation consists of two steps. 

1) To calculate action metrics, use `evaluate.py` script. Usage example:
    
    ```
    python evaluate.py --config configs/wsprites.yaml --run-name test-run --step 470000 --wandb
    ```

2) To calculate image/video quality metrics one first needs to generate evaluation dataset with the help of `build_evaluation_dataset.py` script. Usage example: 

    ```
    python build_evaluation_dataset.py --config configs/wsprites.yaml --run-name test-run --step 470000
    ```
   
    After that the metrics can be calculated using the `evaluate_dataset.py` script. Usage example:

    ```
    python evaluate_dataset.py --config configs/wsprites.yaml
    ```
