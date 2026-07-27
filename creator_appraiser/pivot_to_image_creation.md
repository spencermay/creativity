# Part 2: Image creation instead of digit reconstruction

Without breaking the old code generating MNIST digits, we will create a new Creator/Appraiser pair for text-based image generation to demonstrate that the method has wide applicability.

## Creative image generation.
Rather than produce images of MNIST digits, we will produce images matching a short prompt (first we use ImageNet classifications: e.g. 'lion' or 'golden retriever' or 'volcano', and then we use open-ended prompts like 'alien botanical garden' or 'translucent koi fish swimming in the clouds' or 'stained glass cathedral at night') with a Stable Diffusion v1.5 model as the Creator.

The Appraiser will be a contrastive image-text model CLIP with both image and text encoders frozen, projecting into a shared embedding space where matching image-text pairs have high cosine similarity.
The inner loop fine-tunes a low-rank adapter of rank r on the text projection layer, $W'=W + AB$ with A initialized to zero. The T-step inner loop updates A and B to maximize cos(e_img, e_text), where e_text is the LoRA-adapted text feature; the reward is the similarity improvement R = simT − sim0.
This is the natural CLIP analog of the Autoencoder Appraiser: both fine-tune a small predictor against a target representation, asking how well a small adaptation can match the candidate. Rank r restricts adaptation capacity: at small r, the inner loop cannot trivially memorize any image’s direction in T steps.
With the LoRA adapter, the similarity-improvement reward R alone is sufficient without the need for additional reward shaping.

Unlike the MNIST setting, where the Creator’s learnable variable is the latent noise $z$ and the meta-gradient flows back to $z$ at each denoising step, here the learnable variable is the text embedding: the prompt is encoded once, the encoded representation becomes a learnable parameter, and the meta-gradient updates it at each denoising step. This design choice puts the search in semantic space rather than pixel/latent space.