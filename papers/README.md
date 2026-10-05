# References — C3 Task-driven Log-Mel Feature Compression

All PDFs are stored locally in this folder; technical documentation snapshots are in `tech_docs/`.
Machine-readable tracker: `paper_tracker.csv` (status: downloaded → read → summarized → cited).

## Papers

| ID | Citation | PDF filename | DOI / arXiv / URL | Purpose in project |
|---|---|---|---|---|
| 01 | J. Ballé, V. Laparra, E. P. Simoncelli. *End-to-end Optimized Image Compression.* ICLR 2017. | 01_Balle_2017_End_to_End_Optimized_Image_Compression.pdf | arXiv:1611.01704 | Learned transform coding foundation for the AE codec |
| 02 | L. Theis, W. Shi, A. Cunningham, F. Huszár. *Lossy Image Compression with Compressive Autoencoders.* ICLR 2017. | 02_Theis_2017_Compressive_Autoencoders.pdf | arXiv:1703.00395 | Rounding + straight-through gradient for latent quantization |
| 03 | R. Torfason et al. *Towards Image Understanding from Deep Compression without Decoding.* ICLR 2018. | 03_Torfason_2018_Deep_Compression_Understanding.pdf | arXiv:1803.06131 | Joint compression/task training; motivation for task-driven loss |
| 04 | A. E. Eshratifar, A. Esmaili, M. Pedram. *BottleNet.* ISLPED 2019. | 04_BottleNet_2019.pdf | arXiv:1902.01000 | Device→cloud split with learnable bottleneck (node→gateway setting) |
| 05 | J. Shao, J. Zhang. *BottleNet++.* ICC Workshops 2020. | 05_BottleNetPlusPlus_2020.pdf | arXiv:1910.14315 | End-to-end feature compression for device-edge co-inference |
| 06 | J. Shao, Y. Mao, J. Zhang. *Learning Task-Oriented Communication for Edge Inference: An Information Bottleneck Approach.* IEEE JSAC 2022. | 06_Task_Oriented_Communication_Information_Bottleneck.pdf | arXiv:2102.04170 | Theoretical framing of rate vs task-relevant information |
| 07 | N. Li, A. Iosifidis, Q. Zhang. *Attention-based Feature Compression for CNN Inference Offloading in Edge Computing.* ICC 2023. | 07_Attention_Based_Feature_Compression.pdf | arXiv:2211.13745 | Recent learned feature compression for offloading |
| 08 | O. Iakovenko, I. Bondarenko. *Convolutional Variational Autoencoders for Spectrogram Compression in Automatic Speech Recognition.* 2024. | 08_Spectrogram_Compression_ASR.pdf | arXiv:2410.02560 | Closest prior work: compressing speech spectrograms for a recogniser |
| 09 | Y. Matsubara, R. Yang, M. Levorato, S. Mandt. *Supervised Compression for Resource-Constrained Edge Computing Systems.* WACV 2022. | 09_Supervised_Compression_Split_Computing.pdf | arXiv:2108.11898 | Supervised vs reconstruction-only compression at equal rate |
| 10 | B. Jacob et al. *Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference.* CVPR 2018. | 10_Jacob_2018_Integer_Arithmetic_Only_Quantization.pdf | arXiv:1712.05877 | Fake-quant training, INT8 export |
| 11 | Y. Bengio, N. Léonard, A. Courville. *Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation.* 2013. | 11_Bengio_2013_Straight_Through_Estimator.pdf | arXiv:1308.3432 | Straight-through estimator |
| 12 | R. David et al. *TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems.* MLSys 2021. | 12_David_2021_TensorFlow_Lite_Micro.pdf | arXiv:2010.08678 | ESP32-S3 inference runtime |
| 13 | A. Paszke et al. *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 2019. | 13_Paszke_2019_PyTorch.pdf | arXiv:1912.01703 | Training framework |
| 14 | B. McFee et al. *librosa: Audio and Music Signal Analysis in Python.* SciPy 2015. | 14_McFee_2015_librosa.pdf | doi:10.25080/Majora-7b98e3ed-003 | Log-mel extraction |

## Technical documentation (`tech_docs/`)

| ID | Source | File | URL | Purpose |
|---|---|---|---|---|
| T01 | FSDD README (commit 26eb9aa) | T01_FSDD_README.md | https://github.com/Jakobovski/free-spoken-digit-dataset | Dataset description, file naming |
| T02 | librosa.feature.melspectrogram | T02_librosa_melspectrogram.pdf/.html | https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html | Mel parameters |
| T03 | librosa.filters.mel | T03_librosa_mel_filters.pdf/.html | https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html | Slaney norm, htk flag |
| T04 | torch.nn.CrossEntropyLoss | T04_pytorch_CrossEntropyLoss.pdf | https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html | Classification loss on logits |
| T05 | torch.nn.Conv2d | T05_pytorch_Conv2d.pdf | https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html | Encoder/classifier layers |
| T06 | torch.nn.ConvTranspose2d | T06_pytorch_ConvTranspose2d.pdf | https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html | Decoder upsampling |
| T07 | scipy.fft.dctn | T07_scipy_fft_dctn.pdf/.html | https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dctn.html | DCT-II ortho baseline |
| T08 | scikit-learn Common pitfalls: data leakage | T08_sklearn_common_pitfalls_data_leakage.pdf/.html | https://scikit-learn.org/stable/common_pitfalls.html | Train-only normalization rule |
| T09 | sklearn LeaveOneGroupOut | T09_sklearn_LeaveOneGroupOut.pdf/.html | https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html | Speaker-grouped CV |
| T10 | sklearn f1_score | T10_sklearn_f1_score.pdf/.html | https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html | Macro-F1 definition |

FSDD dataset citation: Z. Jackson et al., *Jakobovski/free-spoken-digit-dataset*, Zenodo, https://zenodo.org/badge/latestdoi/61622039 (version used: see `../data/DATASET_VERSION.txt`).
