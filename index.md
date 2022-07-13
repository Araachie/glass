<style>
  details {
    width: 100%;
    margin: 0 auto ;
    background: rgb(255, 255, 255);
    margin-bottom: .5rem;
    border-radius: 5px;
    overflow: hidden;
  }

  summary {
    padding: 1rem;
    display: block;
    background: rgba(20, 94, 146, 0.8);
    padding-left: 2.2rem;
    position: relative;
    cursor: pointer;
  }
  
  summary > i {
    color: white
  }

  summary:before {
    content: '';
    border-width: .4rem;
    border-style: solid;
    border-color: transparent transparent transparent #fff;
    position: absolute;
    top: 1.3rem;
    left: 1rem;
    transform: rotate(0);
    transform-origin: .2rem 50%;
    transition: .25s transform ease;
  }

  details[open] > summary:before {
    transform: rotate(90deg);
  }

  details summary::-webkit-details-marker {
    display:none;
  }
  
  div.a {
    transform: rotate(90deg);
  }
  
  #two_col {
     width:96%;
     margin:0 auto;
  }

  #left_col {
     float:left;
     width:46%;
  }

  #right_col {
     float:right;
     width:46%;
  }
</style>

----------------------------

<div id="two_col">
  <div id="left_col">
    <p align="center">
      <b style="font-size: 24px">Paper:</b><br>
      <a href="https://arxiv.org/abs/2204.06558" style="font-size: 24px; text-decoration: none">[Arxiv]</a>
    </p>
  </div>

  <div id="right_col">
    <p align="center">
      <b style="font-size: 24px">Supplementary:</b><br>
      <a href="https://arxiv.org/abs/2204.06558" style="font-size: 24px; text-decoration: none">[Arxiv]</a>
    </p>
  </div>
</div>



<p align="center">
  <img src="https://user-images.githubusercontent.com/32042066/178710637-37c5426b-d5e4-45b0-ba8a-d374b08fa9f3.gif">
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
<img src="https://user-images.githubusercontent.com/32042066/178690286-0522fa56-aeb6-472f-843b-58a5594c10de.gif">
<img src="https://user-images.githubusercontent.com/32042066/178690308-43626399-0f8c-44f6-bede-0572e66c4b8b.gif">
</p>

Please, check the paper for the details and the official GitHub repository for the instructions on generating the data.

## Results

GLASS automatically separates the foreground from the background in video sequences and discovers most relevant global and local actions that can be used at inference time to generate diverse videos. Trained GLASS can be used in a variety of applications: from controllable generation to motion transfer. We test our model on the *W-Sprites*, the *Tennis* and the *BAIR* datasets.

<p align="center">
  <b>Global Actions</b><br>
</p>

<details>
  <summary><i>W-Sprites</i></summary>
  <p align="center"><img width=800 src="https://user-images.githubusercontent.com/32042066/178521009-c52694a3-04d3-4ddd-a404-85b3a8733ad3.gif"></p><br>
  <p align="center">Each row starts with the same frame. Each column corresponds to one of the global actions, from left to right: right, left, down, up and stay.</p>
</details>
<details>
  <summary><i>Tennis</i></summary>
  <p align="center"><img width=800 src="https://user-images.githubusercontent.com/32042066/178578377-451b921f-f7e6-4a73-81ef-6f6fed8f57bf.gif"></p><br>
  <p align="center">Each row starts with the same frame. Each column corresponds to one of the global actions, from left to right: right, left, down, up and stay.</p>
</details>
<details>
  <summary><i>BAIR</i></summary>
  <p align="center"><img width=800 src="https://user-images.githubusercontent.com/32042066/178578501-ec41c0b2-28fd-4545-b60c-1fbdf52e616d.gif"></p>
  <p align="center">Each row starts with the same frame. Each column corresponds to one of the global actions, from left to right: right, left, down, up and stay.</p>
</details>

<p align="center">
  <b>Local Actions</b><br>
</p>

<details>
  <summary><i>W-Sprites</i></summary>
  <p align="center"><img width=600 src="https://user-images.githubusercontent.com/32042066/178588885-d05df9af-7320-40bf-8360-a0ac5f3f8a2d.png"></p>
  <p align="center">The local actions learnt by the model can be interpreted as turn front, slash front, spellcast, slash left, turn right, turn left.</p>
</details>
<details>
  <summary><i>Tennis</i></summary>
  <p align="center"><img width=800 src="https://user-images.githubusercontent.com/32042066/178588879-a2d207ef-70e4-4e06-980d-ea8b16692eeb.png"></p>
  <p align="center">The actions capture some small variations of the pose of the tennis player, such as rotation and the distance between the legs.</p>
</details>
<details>
  <summary><i>BAIR</i></summary>
  <p align="center"><img width=600 src="https://user-images.githubusercontent.com/32042066/178588869-6b12f25a-6916-4007-b8c1-b459ea2f9e63.png"></p>
  <p align="center">The actions capture some local deformations of the robot arm, i.e. the state of the manipulator (open / close).</p>
</details>

<p align="center">
  <b>Motion Transfer</b><br>
</p>

<details>
  <summary><i>Tennis</i></summary>
  <p align="center">
    <img src="https://user-images.githubusercontent.com/32042066/178583770-8e7a2158-7f89-4d5c-b0fb-e0da6ddf6293.gif">
    <img src="https://user-images.githubusercontent.com/32042066/178583883-72b8fa54-5245-4908-9c45-39ba2382d817.gif">
    <img src="https://user-images.githubusercontent.com/32042066/178584071-b7411a9a-69d5-42b1-bf82-baa2fdc071f1.gif">
  </p>
  <p align="center">
    <img src="https://user-images.githubusercontent.com/32042066/178584160-7f770121-d2d7-4e28-9d32-6ecfa71c78dc.gif">
    <img src="https://user-images.githubusercontent.com/32042066/178584244-4fb85734-ab35-46e2-b27f-1eac203d3a18.gif">
    <img src="https://user-images.githubusercontent.com/32042066/178584285-a084a4da-1199-4725-ab40-6360c62a5c3f.gif">
  </p>
  <p align="center">
    <img src="https://user-images.githubusercontent.com/32042066/178584376-5062b9aa-d327-4dc1-8abb-bec87ceab90b.gif">
    <img src="https://user-images.githubusercontent.com/32042066/178584510-88a176ab-6289-415c-82ad-524d908cc0ce.gif">
    <img src="https://user-images.githubusercontent.com/32042066/178584560-49ca644e-7a1d-49b2-97bc-30d528b869c7.gif">
  </p>
  <p align="center">
    Row by row: Original videos, reconstruction and motion transfer examples on the Tennis dataset. Note the ability of GLASS to generate very diverse videos from the same initial frame.
  </p>
</details>
<details>
  <summary><i>BAIR</i></summary>
  <p align="center">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178586943-a9cb58a9-bcd1-406e-bbe1-ae47263c1f86.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587017-acd9b915-8072-4110-a1eb-c95558b17dab.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587095-30aea0db-afc8-4491-bdf5-1cec3d9f8a0b.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587226-687d44ef-6a99-483a-bbed-43f11581ab7e.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587297-e4e90b4a-6d0a-4eb1-9172-7d7da06fbdd2.gif">
  </p>
  <p align="center">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587661-db15154d-bffd-49cb-9e96-4d22c727bb33.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587686-5446781e-6fa8-4c9d-98f9-fd098a4bd13e.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587714-a345343d-39a8-49aa-8433-22dc29243483.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587739-4c0447d4-9375-4682-a01d-ee8bfe78ee30.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587757-019541af-d640-4766-9a5f-6b5faaffbeb0.gif">
  </p>
  <p align="center">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587770-f85f5509-3ebf-4bb6-8fc3-1552d30166b5.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587790-fcd2516d-fed0-43e5-b272-4779611afe66.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587810-3113d210-9d66-4fb1-a405-292546a86d95.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587828-10e33d1a-b3f5-4b55-9c77-8c7f37004d85.gif">
    <img width=150 src="https://user-images.githubusercontent.com/32042066/178587850-9a7e440e-5ef7-4bcf-84ba-4e53bba4ba4e.gif">
  </p>
  <p align="center">
    Row by row: Original videos, reconstruction and motion transfer examples on the BAIR dataset. Note the ability of GLASS to generate very diverse videos from the same initial frame.
  </p>
</details>

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
    
