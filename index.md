<style>
.accordion {
  background-color: #eee;
  color: #444;
  cursor: pointer;
  padding: 18px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 15px;
  transition: 0.4s;
}

.active, .accordion:hover {
  background-color: #ccc; 
}

.panel {
  padding: 0 18px;
  display: none;
  background-color: white;
  overflow: hidden;
}
</style>

----------------------------

<p align="center">
  <b style="font-size: 24px">Paper:</b><br>
  <a href="https://arxiv.org/abs/2204.06558" style="font-size: 24px; text-decoration: none">[Arxiv]</a>
</p>

<p align="center">
  <b style="font-size: 24px">Supplementary:</b><br>
  <a href="https://arxiv.org/abs/2204.06558" style="font-size: 24px; text-decoration: none">[Arxiv]</a>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/32042066/178519016-d447fe4a-2d43-4495-baab-82b85de6a30a.png">
</p>

## Abstract

We present GLASS, a method for Global and Local Action-driven Sequence Synthesis. GLASS is a generative model that is trained on video sequences in an unsupervised manner and that can animate an input image at test time. The method learns to segment frames into foreground-background layers and to generate transitions of the foregrounds over time through a global and local action representation. Global actions are explicitly related to 2D shifts, while local actions are instead related to (both geometric and photometric) local deformations. GLASS uses a recurrent neural network to transition between frames and is trained through a reconstruction loss. We also introduce W-Sprites (Walking Sprites), a novel synthetic dataset with a predefined action space. We evaluate our method on both W-Sprites and real datasets, and find that GLASS is able to generate realistic video sequences from a single input image and to successfully learn a more advanced action space than in prior work.

## Method

GLASS consists of two stages: One is the Global Motion Analysis (GMA) and the other is the Local Motion Analysis (LMA). GMA aims to separate the foreground agent from the background and to also regress the 2D shifts between foregrounds and backgrounds. LMA aims to learn a representation for local actions that can describe deformations other than 2D shifts. Towards this purpose it uses a Recurrent Neural Network (RNN) and a feature encoding of a frame and of the global and local actions as input. Both GMA and LMA stages are jointly trained in an unsupervised manner.

<p align="center">
  <b>Global Motion Analysis</b><br>
  <img src="https://user-images.githubusercontent.com/32042066/178463132-8b27e3aa-f084-44a5-b71b-80635e418ee6.png">
</p>

Two input frames <img src="https://latex.codecogs.com/svg.image?I_t"> and <img src="https://latex.codecogs.com/svg.image?I_{t+1}"> are fed (separately) to a segmentation network to output the foreground masks <img src="https://latex.codecogs.com/svg.image?m_t"> and <img src="https://latex.codecogs.com/svg.image?m_{t+1}"> respectively. The masks are used to separate the foregrounds <img src="https://latex.codecogs.com/svg.image?f_t"> and <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> from the backgrounds <img src="https://latex.codecogs.com/svg.image?b_t"> and <img src="https://latex.codecogs.com/svg.image?b_{t+1}">. The concatenated foregrounds are fed to the network Pf to predict their relative shift <img src="https://latex.codecogs.com/svg.image?\Delta_F">. We use <img src="https://latex.codecogs.com/svg.image?\Delta_F"> to shift <img src="https://latex.codecogs.com/svg.image?f_t"> and match it to <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> via an <img src="https://latex.codecogs.com/svg.image?L_2"> loss (foregrounds may not match exactly and this loss does not penalize small errors). In the case of the backgrounds we also train an inpainting network before shifting them with the predicted <img src="https://latex.codecogs.com/svg.image?\Delta_B"> and matching them with an <img src="https://latex.codecogs.com/svg.image?L_1"> loss (unlike foregrounds, we can expect backgrounds to match).

<p align="center">
  <b>Local Motion Analysis</b><br>
  <img src="https://user-images.githubusercontent.com/32042066/178464184-f9e3b721-02be-43fb-83bc-21cae391a18c.png">
</p>

We feed the segmented foreground <img src="https://latex.codecogs.com/svg.image?f_t">, its shifted version and <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> separately as inputs to an encoder network to obtain features <img src="https://latex.codecogs.com/svg.image?\phi_t">, <img src="https://latex.codecogs.com/svg.image?\tilde\phi_t"> and <img src="https://latex.codecogs.com/svg.image?\phi_{t+1}"> respectively. The latter two features are then mapped to an action at by the action network. A further encoding of <img src="https://latex.codecogs.com/svg.image?\phi_t"> into <img src="https://latex.codecogs.com/svg.image?e_t">, the previous state <img src="https://latex.codecogs.com/svg.image?s_t">, and the local action <img src="https://latex.codecogs.com/svg.image?a_t"> and global action <img src="https://latex.codecogs.com/svg.image?\Delta_F"> are fed as input to the RNN to predict the next state <img src="https://latex.codecogs.com/svg.image?s_{t+1}">. Finally, a decoder maps the state <img src="https://latex.codecogs.com/svg.image?s_{t+1}"> to the next foreground <img src="https://latex.codecogs.com/svg.image?\hat&space;f_{t+1}">, which is matched to the original foreground <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> via the reconstruction loss.

## W-Sprites

In order to assess and ablate the components of GLASS, we build a synthetic video dataset of cartoon characters acting on a moving background. We call the dataset *W-Sprites* (for Walking Sprites).

Here we provide some sample videos from the *W-Sprites* dataset:

<p align="center">
<img src="https://user-images.githubusercontent.com/32042066/178506364-16cb985d-97b2-4bfd-ac10-051b936628ad.gif">
<img src="https://user-images.githubusercontent.com/32042066/178506599-9f9d398d-24ee-4d9f-a685-5e4422df9b36.gif">
<img src="https://user-images.githubusercontent.com/32042066/178506752-764e1ee6-bff0-4951-a97d-39b4485ce4f0.gif">
<img src="https://user-images.githubusercontent.com/32042066/178507581-3690cb7b-ced1-4018-a285-7be5d721b1d3.gif">
</p>


## Results

GLASS automatically separates the foreground from the background in video sequences and discovers most relevant global and local actions that can be used at inference time to generate diverse videos. Trained GLASS can be used in a variety of applications: from controllable generation to motion transfer. We test our model on the *W-Sprites*, the *Tennis* and the *BAIR* datasets.

<p align="center">
  <b>Global Actions</b><br>
</p>
<button class="accordion">W-Sprites</button>
<div class="panel">
  <p align="center"><img src="https://user-images.githubusercontent.com/32042066/178521009-c52694a3-04d3-4ddd-a404-85b3a8733ad3.gif"></p>
</div>
<button class="accordion"><i>Tennis</i></button>
<div class="panel">
  <p align="center"><img src="https://user-images.githubusercontent.com/32042066/178527099-b0ca39a0-72c0-4de4-aabc-d8a4575b3e3b.gif"></p>
</div>
<button class="accordion"><i>BAIR</i></button>
<div class="panel">
  <p align="center"><img src="https://user-images.githubusercontent.com/32042066/178527406-b146ba7f-74c2-433f-9a6e-d033ac2390a4.gif"></p>
</div>

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
    
<script>
var acc = document.getElementsByClassName("accordion");
var i;

for (i = 0; i < acc.length; i++) {
  acc[i].addEventListener("click", function() {
    this.classList.toggle("active");
    var panel = this.nextElementSibling;
    if (panel.style.display === "block") {
      panel.style.display = "none";
    } else {
      panel.style.display = "block";
    }
  });
}
</script>
