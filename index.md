----------------------------

<p align="center">
  <b style="font-size: 24px">Paper:</b><br>
  <a href="https://arxiv.org/abs/2204.06558" style="font-size: 24px; text-decoration: none">[Arxiv]</a>
</p>

<h2>Abstract</h2>

We present GLASS, a method for Global and Local Action-driven Sequence Synthesis. GLASS is a generative model that is trained on video sequences in an unsupervised manner and that can animate an input image at test time. The method learns to segment frames into foreground-background layers and to generate transitions of the foregrounds over time through a global and local action representation. Global actions are explicitly related to 2D shifts, while local actions are instead related to (both geometric and photometric) local deformations. GLASS uses a recurrent neural network to transition between frames and is trained through a reconstruction loss. We also introduce W-Sprites (Walking Sprites), a novel synthetic dataset with a predefined action space. We evaluate our method on both W-Sprites and real datasets, and find that GLASS is able to generate realistic video sequences from a single input image and to successfully learn a more advanced action space than in prior work.

<h2>Method</h2>

GLASS consists of two stages: One is the Global Motion Analysis (GMA) and the other is the Local Motion Analysis (LMA). GMA aims to separate the foreground agent from the background and to also regress the 2D shifts between foregrounds and backgrounds. LMA aims to learn a representation for local actions that can describe deformations other than 2D shifts. Towards this purpose it uses a Recurrent Neural Network (RNN) and a feature encoding of a frame and of the global and local actions as input. Both GMA and LMA stages are jointly trained in an unsupervised manner.

<p align="center">
<b>Global Motion Analysis</b>
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/32042066/178463132-8b27e3aa-f084-44a5-b71b-80635e418ee6.png">
</p>

Two input frames <img src="https://latex.codecogs.com/svg.image?I_t"> and <img src="https://latex.codecogs.com/svg.image?I_{t+1}"> are fed (separately) to a segmentation network to output the foreground masks <img src="https://latex.codecogs.com/svg.image?m_t"> and <img src="https://latex.codecogs.com/svg.image?m_{t+1}"> respectively. The masks are used to separate the foregrounds <img src="https://latex.codecogs.com/svg.image?f_t"> and <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> from the backgrounds <img src="https://latex.codecogs.com/svg.image?b_t"> and <img src="https://latex.codecogs.com/svg.image?b_{t+1}">. The concatenated foregrounds are fed to the network Pf to predict their relative shift <img src="https://latex.codecogs.com/svg.image?\Delta_F">. We use <img src="https://latex.codecogs.com/svg.image?\Delta_F"> to shift <img src="https://latex.codecogs.com/svg.image?f_t"> and match it to <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> via an <img src="https://latex.codecogs.com/svg.image?L_2"> loss (foregrounds may not match exactly and this loss does not penalize small errors). In the case of the backgrounds we also train an inpainting network before shifting them with the predicted <img src="https://latex.codecogs.com/svg.image?\Delta_B"> and matching them with an <img src="https://latex.codecogs.com/svg.image?L_1"> loss (unlike foregrounds, we can expect backgrounds to match).

<p align="center">
<b>Local Motion Analysis</b>
</p>
<p align="center">
<img src="https://user-images.githubusercontent.com/32042066/178464184-f9e3b721-02be-43fb-83bc-21cae391a18c.png">
</p>

We feed the segmented foreground <img src="https://latex.codecogs.com/svg.image?f_t">, its shifted version and <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> separately as inputs to an encoder network to obtain features <img src="https://latex.codecogs.com/svg.image?\phi_t">, <img src="https://latex.codecogs.com/svg.image?\tilde\phi_t"> and <img src="https://latex.codecogs.com/svg.image?\phi_{t+1}"> respectively. The latter two features are then mapped to an action at by the action network. A further encoding of <img src="https://latex.codecogs.com/svg.image?\phi_t"> into <img src="https://latex.codecogs.com/svg.image?e_t">, the previous state <img src="https://latex.codecogs.com/svg.image?s_t">, and the local action <img src="https://latex.codecogs.com/svg.image?a_t"> and global action <img src="https://latex.codecogs.com/svg.image?\Delta_F"> are fed as input to the RNN to predict the next state <img src="https://latex.codecogs.com/svg.image?s_{t+1}">. Finally, a decoder maps the state <img src="https://latex.codecogs.com/svg.image?s_{t+1}"> to the next foreground <img src="https://latex.codecogs.com/svg.image?\hat&space;f_{t+1}">, which is matched to the original foreground <img src="https://latex.codecogs.com/svg.image?f_{t+1}"> via the reconstruction loss.

### Results

...
