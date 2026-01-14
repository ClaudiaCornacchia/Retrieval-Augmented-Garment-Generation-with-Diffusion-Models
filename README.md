# Retrieval-Augmented-Garment-Generation-with-Diffusion-Models
A resource-efficient adaptation of the RAGDiffusion framework for Google Colab, designed to synthesize faithful flat-lay garment images from in-the-wild inputs and mitigate structural hallucinations by retrieving external geometric and semantic priors

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ClaudiaCornacchia/Retrieval-Augmented-Garment-Generation-with-Diffusion-Models/blob/main/Retrieval-Augmented%20Garment%20Generation%20with%20Diffusion%20Models.ipynb)

**Author:** Claudia Cornacchia

## Overview
This project is a replication and adaptation of the **RAGDiffusion** framework [1], designed to function within the computational constraints of **Google Colab**.

The goal is to solve "Structural Hallucination" in garment generation by using a **Retrieval-Augmented Generation (RAG)** pipeline. Instead of relying solely on text prompts, the model retrieves structurally similar garments from a database to guide the diffusion process.

## Key Engineering Adaptations
To adapt the original architecture for limited hardware, I implemented the following optimizations:

### 1. Retrieval Database
- **Problem:** Searching the full VITON-HD dataset (13k images) was memory-prohibitive.
- **Solution:** Implemented **DBSCAN clustering** and **Cluster-Aware Sampling** to reduce the retrieval database to **2,000 prototypes** while preserving structural diversity.

### 2. Geometric Guidance (Hard Constraint)
- **Adaptation:** Replaced the original paper's custom "Landmark Guider" (which requires training from scratch) with **ControlNet-Canny (SDXL)**.
- **Method:** The retrieved binary garment silhouette is used as a hard edge map constraint, forcing the diffusion model to respect the physical shape of the reference.

### 3. Cohesion-Based Prototype Selection
- **Improvement:** The original **SLLE** averaging method produced blurry "ghost" masks.
- **My Approach:** Implemented a **Cohesion-Based Selection** algorithm that calculates IoU overlap among neighbors and selects the single most representative prototype, eliminating artifacts.


## 📊 Results
Due to compute limitations, the **EP-Adapter** weights remain untrained, meaning texture transfer is limited. However, the pipeline successfully validates the geometric control mechanisms and the overall validity of the multi level conditioning pipeline.

| Metric | Score | Insight |
|:---:|:---:|:---|
| **LPIPS** | **0.706** | Macro-structures (logos, shape) are quite preserved; fine textures suffer due to the untrained adapter. |
| **Recall@1** | **0.85** | StructureNet successfully aligns "In-the-Wild" images with "Flat-Lay" products. |
| **Mean IoU** | **0.82**| Good overall geometric overlap between the retrieved silhouette and the input image.|


## 💻 How to Run
1. Click the **"Open in Colab"** badge above.
2. The notebook handles dependency installation automatically.
3. **Note:** Dataset paths in the code are set to relative directories.

## 📚 References

[1] Y. Li, X. Tan, W. Shang, Y. Wu, J. Wang, X. Chen, Y. Zhang, R. Lin, and B. Ni, "RAG-Diffusion: Faithful Cloth Generation via External Knowledge Assimilation," *arXiv preprint arXiv:2411.19528v2*, 2025. (Primary architectural framework).

[2] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach, "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis," *arXiv preprint arXiv:2307.01952*, 2023. (Backbone generative model).

[3] J. Bai, S. Bai, S. Yang, S. Wang, S. Tan, P. Wang, J. Lin, C. Zhou, and J. Zhou, "Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond," *arXiv preprint arXiv:2308.12966*, 2023. (Used for semantic attribute extraction).

[4] H. Ye, J. Zhang, S. Liu, X. Han, and W. Yang, "IP-Adapter: Text Compatible Image Prompt Adapter for Text-to-Image Diffusion Models," *arXiv preprint arXiv:2308.06721*, 2023. (Basis for EP-Adapter architecture).

[5] L. Hu, X. Gao, P. Zhang, K. Sun, B. Zhang, and L. Bo, "Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation," *arXiv preprint arXiv:2311.17117*, 2023. (Methodology for ReferenceNet).

[6] S. Choi, S. Park, M. Lee, and J. Choo, "VITON-HD: High-Resolution Virtual Try-On via Misalignment-Aware Normalization," *arXiv preprint arXiv:2103.16874*, 2021. (Dataset source).  
