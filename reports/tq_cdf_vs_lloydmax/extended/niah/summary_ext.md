# Extended TurboQuant ablation: uniform grids and 5-bit scaling

CPU synthetic evidence only; no CAModel or device latency claims.

128 coordinates, Pi = D H D, one FP16 scale per vector or group; no FP8 intermediate. NIAH uses uniform 4:1 CSA reduction, 16 needles and 16 paired adversarial distractors (pre-compression cosine 0.70–0.82). Rank is 1-based against every compressed slot; ties prefer the smaller slot ID. FPR is the fraction of the 16 paired distractors in top-128.

Synthetic inputs are Gaussian with pre-rotation gains on channels 0,16,64,96. Synthetic retrieval fields are n/a. NIAH uses anisotropic unit vectors with gain 8. Spearman uses sampled backgrounds plus all 32 special slots. Each seed is one paired observation; needles are averaged within a seed. Contexts and kappa strata are never pooled for t tests.

GaussianCDF evaluates ndtr/ndtri. Sigmoid-Vasicek evaluates expit(1.5976*z + 0.070565*z^3) and logit plus three Newton iterations at every reconstructed coordinate. The requested label identifies this cubic logistic fit; no claim of historical attribution is made. Analytical paths do not use reconstruction lookup tables.

Unflattened Uniform-Midrise stores s=absmax/(2^(b-1)-0.5), codes floor(v/s+2^(b-1)), and reconstructs (code-2^(b-1)+0.5)*s. Uniform-Midtread stores s=absmax/(2^(b-1)-1), rounds to nearest signed integer (ties to even), clips to [-2^(b-1),2^(b-1)-1], and reconstructs q*s. Its symmetric absmax convention normally leaves the most negative code unused. Both normalize with the stored FP16 step and use no CDF. The other three methods store FP16 RMS. Scale storage overhead is identical; scale estimation rules differ intentionally.

The 3/4-bit Lloyd-Max reference retains its original 256 updates. The 5-bit baseline solves the analytic 32-cell Gaussian conditional-mean equations to convergence (FP64 update <1e-14), then stores FP32 centres and thresholds; it is not a slice of a smaller codebook.

NMSE(key) is sum(||K-Khat||^2)/sum(||K||^2), preserving the previous nmse CSV column. NMSE(QK) is sum((QK^T-QKhat^T)^2)/sum((QK^T)^2), recorded as logit_nmse. Both are evaluated across the full streamed cache; only Spearman is sampled.

RoPE is a separate clean 64-channel half-mode probe, theta=160000. Phase error is the requested full-carrier acos angle, not an estimate of each pair's phase. Logit share follows DSV4's |rotary|/(|rotary|+|semantic|) convention. The counterfactual carrier is quantized; NIAH retrieval scores use content only.

stream_latency_s includes shared data generation/rotation/exact scores and the method's encode/pack/decode/score/rank/sample work. codec_us_per_slot isolates encode/pack/decode. Setup, warmup, final correlation/RoPE and diagnostic reductions are excluded. Method order is shuffled per chunk. Latency is host NumPy/SciPy wall time.

levels_used counts distinct decoded coordinates per vector (different group scales may produce more than 2^b values). code_levels_used in runs.csv counts distinct integer codes.

Seeds: 11,23,37,53,71; group size: per vector; chunk tokens: 8192; BLAS threads: 1.

## Mean metrics across seeds

| Source | Context / kappa | Bits | Regime | Recall@128 | Exact recall | Rank | Distractor FPR | Rho | Dilution | NMSE(key) | NMSE(QK) | Cosine | Levels |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| niah | 8192 | 3 | TQ-LloydMax | 0.55 | 0.5625 | 221.55 | 0.475 | 0.997895 | 0.993782 | 0.0312266 | 0.00340911 | 0.98434 | 7.99229 |
| niah | 8192 | 3 | TQ-GaussianCDF | 0.5375 | 0.5625 | 230.213 | 0.4625 | 0.99695 | 1.00686 | 0.0454373 | 0.0075872 | 0.977027 | 8 |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | 0.5375 | 0.5625 | 229.963 | 0.4625 | 0.99695 | 1.00935 | 0.045394 | 0.00754735 | 0.97704 | 8 |
| niah | 8192 | 3 | TQ-Uniform-Midrise | 0.55 | 0.5625 | 226.875 | 0.5 | 0.996946 | 0.829687 | 0.0456034 | 0.00452672 | 0.977532 | 7.57646 |
| niah | 8192 | 3 | TQ-Uniform-Midtread | 0.5375 | 0.5625 | 225.15 | 0.4625 | 0.995843 | 1.10694 | 0.0620814 | 0.0061683 | 0.96976 | 6.66602 |
| niah | 8192 | 4 | TQ-LloydMax | 0.5625 | 0.5625 | 223.675 | 0.475 | 0.999413 | 1.04575 | 0.00867781 | 0.000875837 | 0.995683 | 14.922 |
| niah | 8192 | 4 | TQ-GaussianCDF | 0.5625 | 0.5625 | 226.688 | 0.4625 | 0.998933 | 0.969845 | 0.0162001 | 0.0019422 | 0.991697 | 15.9944 |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | 0.5625 | 0.5625 | 226.6 | 0.4625 | 0.998933 | 0.971007 | 0.0161727 | 0.00193955 | 0.991711 | 15.9946 |
| niah | 8192 | 4 | TQ-Uniform-Midrise | 0.5625 | 0.5625 | 221.088 | 0.475 | 0.99931 | 0.922236 | 0.00991858 | 0.000989857 | 0.994965 | 14.5495 |
| niah | 8192 | 4 | TQ-Uniform-Midtread | 0.575 | 0.5625 | 221.8 | 0.475 | 0.999223 | 0.990365 | 0.0113968 | 0.00112673 | 0.994218 | 13.7173 |
| niah | 8192 | 5 | TQ-LloydMax | 0.5625 | 0.5625 | 222.625 | 0.475 | 0.999837 | 0.988851 | 0.00229854 | 0.000233545 | 0.99886 | 27.415 |
| niah | 8192 | 5 | TQ-GaussianCDF | 0.5625 | 0.5625 | 224.325 | 0.475 | 0.999615 | 0.998408 | 0.00573783 | 0.000608158 | 0.997011 | 31.4324 |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | 0.5625 | 0.5625 | 224.2 | 0.475 | 0.999615 | 0.99839 | 0.00572856 | 0.000609126 | 0.997015 | 31.433 |
| niah | 8192 | 5 | TQ-Uniform-Midrise | 0.5625 | 0.5625 | 222.713 | 0.4625 | 0.999833 | 1.01155 | 0.0023237 | 0.000229058 | 0.998812 | 26.7884 |
| niah | 8192 | 5 | TQ-Uniform-Midtread | 0.5625 | 0.5625 | 222.95 | 0.4875 | 0.99982 | 0.972973 | 0.00247877 | 0.000248249 | 0.998733 | 26.0807 |
| niah | 32768 | 3 | TQ-LloydMax | 0.3625 | 0.4125 | 858.362 | 0.3 | 0.997927 | 0.926329 | 0.0312465 | 0.00340958 | 0.984337 | 7.99136 |
| niah | 32768 | 3 | TQ-GaussianCDF | 0.4 | 0.4125 | 890.288 | 0.2625 | 0.996992 | 0.98484 | 0.0454631 | 0.0075909 | 0.977014 | 7.99998 |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | 0.4 | 0.4125 | 890.087 | 0.2625 | 0.996993 | 0.984815 | 0.0454198 | 0.00755348 | 0.977027 | 7.99998 |
| niah | 32768 | 3 | TQ-Uniform-Midrise | 0.4125 | 0.4125 | 858.45 | 0.2875 | 0.996996 | 0.933492 | 0.0456847 | 0.00457404 | 0.977481 | 7.57339 |
| niah | 32768 | 3 | TQ-Uniform-Midtread | 0.3875 | 0.4125 | 863.775 | 0.275 | 0.995906 | 0.999006 | 0.0622006 | 0.00621982 | 0.96972 | 6.66206 |
| niah | 32768 | 4 | TQ-LloydMax | 0.4 | 0.4125 | 847.663 | 0.275 | 0.999431 | 0.986544 | 0.0086919 | 0.000891271 | 0.995675 | 14.919 |
| niah | 32768 | 4 | TQ-GaussianCDF | 0.3875 | 0.4125 | 865.95 | 0.275 | 0.998952 | 0.994694 | 0.0162325 | 0.00195072 | 0.991679 | 15.9947 |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | 0.3875 | 0.4125 | 866.925 | 0.275 | 0.998953 | 0.994458 | 0.0162037 | 0.00194905 | 0.991694 | 15.9947 |
| niah | 32768 | 4 | TQ-Uniform-Midrise | 0.4125 | 0.4125 | 845.875 | 0.3 | 0.999325 | 0.989422 | 0.00994373 | 0.000998093 | 0.994954 | 14.5341 |
| niah | 32768 | 4 | TQ-Uniform-Midtread | 0.4125 | 0.4125 | 861.212 | 0.2875 | 0.999232 | 0.993615 | 0.0114172 | 0.00114108 | 0.99421 | 13.7 |
| niah | 32768 | 5 | TQ-LloydMax | 0.4125 | 0.4125 | 851.438 | 0.2875 | 0.999846 | 1.00233 | 0.00229783 | 0.000232736 | 0.99886 | 27.4146 |
| niah | 32768 | 5 | TQ-GaussianCDF | 0.3875 | 0.4125 | 855.612 | 0.275 | 0.999625 | 1.00564 | 0.00575111 | 0.000613562 | 0.997005 | 31.4401 |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | 0.3875 | 0.4125 | 855.875 | 0.275 | 0.999626 | 1.00681 | 0.00574235 | 0.000614343 | 0.997009 | 31.4393 |
| niah | 32768 | 5 | TQ-Uniform-Midrise | 0.4125 | 0.4125 | 851.35 | 0.3 | 0.999839 | 0.998255 | 0.00232858 | 0.000233654 | 0.99881 | 26.7664 |
| niah | 32768 | 5 | TQ-Uniform-Midtread | 0.4125 | 0.4125 | 849.638 | 0.2875 | 0.999827 | 0.992994 | 0.00248413 | 0.000249449 | 0.998731 | 26.0615 |
| niah | 131072 | 3 | TQ-LloydMax | 0.2 | 0.1875 | 3038.4 | 0.1 | 0.997942 | 0.98585 | 0.031222 | 0.00340895 | 0.984349 | 7.99203 |
| niah | 131072 | 3 | TQ-GaussianCDF | 0.175 | 0.1875 | 3103.03 | 0.075 | 0.997017 | 0.977823 | 0.0454036 | 0.00756078 | 0.977034 | 7.99998 |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | 0.175 | 0.1875 | 3102.15 | 0.075 | 0.997019 | 0.977935 | 0.0453603 | 0.00752367 | 0.977048 | 7.99998 |
| niah | 131072 | 3 | TQ-Uniform-Midrise | 0.1875 | 0.1875 | 2986.59 | 0.1125 | 0.996968 | 0.977342 | 0.0456307 | 0.00458042 | 0.977495 | 7.57593 |
| niah | 131072 | 3 | TQ-Uniform-Midtread | 0.2 | 0.1875 | 2957.25 | 0.0875 | 0.995928 | 0.977237 | 0.0621316 | 0.00623767 | 0.969758 | 6.66447 |
| niah | 131072 | 4 | TQ-LloydMax | 0.1875 | 0.1875 | 2968.01 | 0.0875 | 0.999441 | 0.997847 | 0.00868175 | 0.000892498 | 0.995681 | 14.9173 |
| niah | 131072 | 4 | TQ-GaussianCDF | 0.2 | 0.1875 | 3031.11 | 0.075 | 0.998962 | 0.999415 | 0.0161931 | 0.00194387 | 0.991694 | 15.9951 |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | 0.2 | 0.1875 | 3031.36 | 0.075 | 0.998964 | 0.999406 | 0.0161648 | 0.00194096 | 0.991709 | 15.995 |
| niah | 131072 | 4 | TQ-Uniform-Midrise | 0.1875 | 0.1875 | 2973.31 | 0.1125 | 0.999327 | 0.996121 | 0.00993023 | 0.000995775 | 0.994959 | 14.5365 |
| niah | 131072 | 4 | TQ-Uniform-Midtread | 0.1875 | 0.1875 | 2993.96 | 0.0875 | 0.999234 | 0.994608 | 0.0114041 | 0.00114161 | 0.994216 | 13.7016 |
| niah | 131072 | 5 | TQ-LloydMax | 0.1875 | 0.1875 | 2960.66 | 0.1125 | 0.999847 | 0.998399 | 0.00229684 | 0.000232591 | 0.99886 | 27.4153 |
| niah | 131072 | 5 | TQ-GaussianCDF | 0.1875 | 0.1875 | 3008.21 | 0.1 | 0.999626 | 1.00092 | 0.00572815 | 0.000612039 | 0.997015 | 31.4351 |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | 0.1875 | 0.1875 | 3012.75 | 0.1 | 0.999626 | 0.999431 | 0.00571931 | 0.000612358 | 0.997019 | 31.4349 |
| niah | 131072 | 5 | TQ-Uniform-Midrise | 0.1875 | 0.1875 | 2969.11 | 0.1125 | 0.99984 | 0.998577 | 0.00232431 | 0.000233191 | 0.998812 | 26.7792 |
| niah | 131072 | 5 | TQ-Uniform-Midtread | 0.2 | 0.1875 | 2980.28 | 0.1 | 0.99983 | 0.998624 | 0.00248135 | 0.000249122 | 0.998732 | 26.069 |

## RoPE and timing means

| Source | Context / kappa | Bits | Regime | RoPE share | Phase error deg | Stream s | us/slot | Codec us/slot | Effective bits/coord |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| niah | 8192 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.123381 | 60.2447 | 7.67535 | 3.125 |
| niah | 8192 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.129244 | 63.1076 | 10.6711 | 3.125 |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.13518 | 66.0059 | 13.5593 | 3.125 |
| niah | 8192 | 3 | TQ-Uniform-Midrise | 0.436142 | 11.8961 | 0.119229 | 58.2174 | 5.77567 | 3.125 |
| niah | 8192 | 3 | TQ-Uniform-Midtread | 0.436142 | 13.7418 | 0.11854 | 57.881 | 5.35279 | 3.125 |
| niah | 8192 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.125693 | 61.3733 | 8.9911 | 4.125 |
| niah | 8192 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.128702 | 62.8427 | 10.4633 | 4.125 |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.136715 | 66.7553 | 14.3925 | 4.125 |
| niah | 8192 | 4 | TQ-Uniform-Midrise | 0.436142 | 5.6 | 0.121773 | 59.4593 | 6.94029 | 4.125 |
| niah | 8192 | 4 | TQ-Uniform-Midtread | 0.436142 | 5.98984 | 0.12017 | 58.6768 | 6.19186 | 4.125 |
| niah | 8192 | 5 | TQ-LloydMax | 0.436142 | 2.73921 | 0.128401 | 62.6959 | 10.2515 | 5.125 |
| niah | 8192 | 5 | TQ-GaussianCDF | 0.436142 | 4.70046 | 0.129291 | 63.1301 | 10.7741 | 5.125 |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | 0.436142 | 4.69729 | 0.13889 | 67.8174 | 15.4197 | 5.125 |
| niah | 8192 | 5 | TQ-Uniform-Midrise | 0.436142 | 2.71605 | 0.121902 | 59.5224 | 6.92496 | 5.125 |
| niah | 8192 | 5 | TQ-Uniform-Midtread | 0.436142 | 2.8164 | 0.121404 | 59.2794 | 6.87482 | 5.125 |
| niah | 32768 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.490132 | 59.8306 | 7.49666 | 3.125 |
| niah | 32768 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.512034 | 62.5041 | 10.1104 | 3.125 |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.548236 | 66.9233 | 14.486 | 3.125 |
| niah | 32768 | 3 | TQ-Uniform-Midrise | 0.436142 | 11.8961 | 0.473436 | 57.7924 | 5.39029 | 3.125 |
| niah | 32768 | 3 | TQ-Uniform-Midtread | 0.436142 | 13.7418 | 0.476035 | 58.1097 | 5.77542 | 3.125 |
| niah | 32768 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.504401 | 61.5724 | 9.26527 | 4.125 |
| niah | 32768 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.519345 | 63.3966 | 11.039 | 4.125 |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.546502 | 66.7117 | 14.3415 | 4.125 |
| niah | 32768 | 4 | TQ-Uniform-Midrise | 0.436142 | 5.6 | 0.479281 | 58.506 | 6.12368 | 4.125 |
| niah | 32768 | 4 | TQ-Uniform-Midtread | 0.436142 | 5.98984 | 0.478597 | 58.4225 | 6.09816 | 4.125 |
| niah | 32768 | 5 | TQ-LloydMax | 0.436142 | 2.73921 | 0.514547 | 62.8109 | 10.4622 | 5.125 |
| niah | 32768 | 5 | TQ-GaussianCDF | 0.436142 | 4.70046 | 0.521943 | 63.7138 | 11.3603 | 5.125 |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | 0.436142 | 4.69729 | 0.557455 | 68.0487 | 15.6316 | 5.125 |
| niah | 32768 | 5 | TQ-Uniform-Midrise | 0.436142 | 2.71605 | 0.481244 | 58.7456 | 6.41009 | 5.125 |
| niah | 32768 | 5 | TQ-Uniform-Midtread | 0.436142 | 2.8164 | 0.482537 | 58.9035 | 6.52789 | 5.125 |
| niah | 131072 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 1.95772 | 59.7449 | 7.69093 | 3.125 |
| niah | 131072 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 2.03969 | 62.2463 | 10.1903 | 3.125 |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 2.15239 | 65.6857 | 13.6295 | 3.125 |
| niah | 131072 | 3 | TQ-Uniform-Midrise | 0.436142 | 11.8961 | 1.88491 | 57.5228 | 5.50898 | 3.125 |
| niah | 131072 | 3 | TQ-Uniform-Midtread | 0.436142 | 13.7418 | 1.88929 | 57.6565 | 5.6237 | 3.125 |
| niah | 131072 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 2.009 | 61.3098 | 9.25235 | 4.125 |
| niah | 131072 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 2.06856 | 63.1274 | 11.0751 | 4.125 |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 2.19584 | 67.0117 | 14.8663 | 4.125 |
| niah | 131072 | 4 | TQ-Uniform-Midrise | 0.436142 | 5.6 | 1.91005 | 58.2901 | 6.20192 | 4.125 |
| niah | 131072 | 4 | TQ-Uniform-Midtread | 0.436142 | 5.98984 | 1.90234 | 58.0547 | 6.01376 | 4.125 |
| niah | 131072 | 5 | TQ-LloydMax | 0.436142 | 2.73921 | 2.04755 | 62.4862 | 10.4332 | 5.125 |
| niah | 131072 | 5 | TQ-GaussianCDF | 0.436142 | 4.70046 | 2.07641 | 63.3669 | 11.3105 | 5.125 |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | 0.436142 | 4.69729 | 2.20921 | 67.4197 | 15.3312 | 5.125 |
| niah | 131072 | 5 | TQ-Uniform-Midrise | 0.436142 | 2.71605 | 1.92399 | 58.7156 | 6.6355 | 5.125 |
| niah | 131072 | 5 | TQ-Uniform-Midtread | 0.436142 | 2.8164 | 1.92188 | 58.6512 | 6.62147 | 5.125 |

## Paired differences from Lloyd-Max

Two-sided paired t tests, unadjusted exploratory p-values; 95% paired CIs. Positive deltas mean the candidate metric is larger. Full statistics for every metric are in paired.csv. Undefined t tests are reported as n/a with their status.

| Source | Context / kappa | Bits | Regime | Metric | n | Delta | t | p | 95% CI | Status |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| niah | 8192 | 3 | TQ-GaussianCDF | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000944871 | -8.078 | 0.00127591 | [-0.00126963, -0.000620115] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0142107 | 95.425 | 7.23079e-08 | [0.0137973, 0.0146242] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | logit_nmse | 5 | 0.0041781 | 64.7254 | 3.41319e-07 | [0.00399887, 0.00435732] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.99573 | 3.98822 | 0.016289 | [0.91022, 5.08124] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000944908 | -8.10691 | 0.00125867 | [-0.00126852, -0.000621297] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0141674 | 94.9925 | 7.3633e-08 | [0.0137533, 0.0145815] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00413825 | 66.4453 | 3.07354e-07 | [0.00396533, 0.00431116] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.88396 | 4.83488 | 0.00843135 | [2.50508, 9.26285] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 3 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000948565 | -7.50432 | 0.0016872 | [-0.00129952, -0.000597616] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midrise | nmse | 5 | 0.0143768 | 122.612 | 2.65353e-08 | [0.0140513, 0.0147024] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.00111762 | 33.7592 | 4.5925e-06 | [0.0010257, 0.00120953] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -1.89968 | -3.85926 | 0.0181586 | [-3.26635, -0.533005] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midtread | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.00205167 | -7.91232 | 0.00138054 | [-0.0027716, -0.00133173] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midtread | nmse | 5 | 0.0308548 | 158.399 | 9.52864e-09 | [0.030314, 0.0313956] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.0027592 | 52.2464 | 8.03281e-07 | [0.00261257, 0.00290583] | ok |
| niah | 8192 | 3 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.32256 | -5.59935 | 0.00499423 | [-3.4742, -1.17092] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000479976 | -7.31284 | 0.00186002 | [-0.000662207, -0.000297745] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00752225 | 82.8143 | 1.27441e-07 | [0.00727005, 0.00777444] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | logit_nmse | 5 | 0.00106636 | 40.0111 | 2.33142e-06 | [0.000992368, 0.00114036] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.47225 | 5.16269 | 0.00668562 | [0.680486, 2.26401] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000479912 | -7.27233 | 0.00189934 | [-0.000663134, -0.00029669] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00749488 | 82.3866 | 1.30106e-07 | [0.0072423, 0.00774746] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00106371 | 39.0185 | 2.57732e-06 | [0.000988023, 0.0011394] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.40139 | 12.9006 | 0.000208217 | [4.23891, 6.56387] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 4 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000102558 | -6.28956 | 0.00326444 | [-0.000147831, -5.72851e-05] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midrise | nmse | 5 | 0.00124077 | 29.8737 | 7.47749e-06 | [0.00112545, 0.00135609] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.000114019 | 13.5489 | 0.000171759 | [9.06545e-05, 0.000137384] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -2.05081 | -2.02117 | 0.113346 | [-4.86797, 0.766348] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.00018985 | -9.69685 | 0.000633065 | [-0.000244208, -0.000135491] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midtread | nmse | 5 | 0.00271896 | 67.0948 | 2.95632e-07 | [0.00260645, 0.00283148] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.00025089 | 26.5182 | 1.2019e-05 | [0.000224622, 0.000277158] | ok |
| niah | 8192 | 4 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.79925 | -4.19152 | 0.0137896 | [-4.65346, -0.945037] | ok |
| niah | 8192 | 5 | TQ-GaussianCDF | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 5 | TQ-GaussianCDF | spearman_rho | 5 | -0.0002216 | -7.09433 | 0.00208479 | [-0.000308326, -0.000134875] | ok |
| niah | 8192 | 5 | TQ-GaussianCDF | nmse | 5 | 0.0034393 | 75.8176 | 1.81371e-07 | [0.00331335, 0.00356524] | ok |
| niah | 8192 | 5 | TQ-GaussianCDF | logit_nmse | 5 | 0.000374613 | 42.3194 | 1.8637e-06 | [0.000350036, 0.000399191] | ok |
| niah | 8192 | 5 | TQ-GaussianCDF | codec_us_per_slot | 5 | 0.522568 | 1.70738 | 0.162941 | [-0.327205, 1.37234] | ok |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000221746 | -7.09163 | 0.00208776 | [-0.000308561, -0.00013493] | ok |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00343002 | 75.0729 | 1.88671e-07 | [0.00330317, 0.00355688] | ok |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000375581 | 41.1234 | 2.08972e-06 | [0.000350223, 0.000400938] | ok |
| niah | 8192 | 5 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.16822 | 6.48427 | 0.00291611 | [2.95528, 7.38116] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 5 | TQ-Uniform-Midrise | spearman_rho | 5 | -4.02331e-06 | -5.02308 | 0.00736935 | [-6.24715e-06, -1.79948e-06] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midrise | nmse | 5 | 2.51645e-05 | 4.19353 | 0.0137673 | [8.50361e-06, 4.18253e-05] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midrise | logit_nmse | 5 | -4.48708e-06 | -3.24026 | 0.0316661 | [-8.33188e-06, -6.4229e-07] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -3.32656 | -10.7162 | 0.000429728 | [-4.18844, -2.46468] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 5 | TQ-Uniform-Midtread | spearman_rho | 5 | -1.71669e-05 | -6.90923 | 0.00230192 | [-2.40653e-05, -1.02684e-05] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midtread | nmse | 5 | 0.00018024 | 32.3987 | 5.41111e-06 | [0.000164794, 0.000195686] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midtread | logit_nmse | 5 | 1.47043e-05 | 80.3624 | 1.43711e-07 | [1.41963e-05, 1.52124e-05] | ok |
| niah | 8192 | 5 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -3.3767 | -17.8864 | 5.74203e-05 | [-3.90085, -2.85254] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | recall_at_128 | 5 | 0.0375 | 1.5 | 0.208 | [-0.0319111, 0.106911] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000934733 | -7.09809 | 0.00208064 | [-0.00130036, -0.000569109] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0142166 | 61.6215 | 4.15394e-07 | [0.0135761, 0.0148572] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | logit_nmse | 5 | 0.00418132 | 45.6251 | 1.38021e-06 | [0.00392687, 0.00443577] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.61371 | 26.347 | 1.2333e-05 | [2.33827, 2.88914] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0.0375 | 1.5 | 0.208 | [-0.0319111, 0.106911] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000933371 | -7.1361 | 0.00203933 | [-0.00129652, -0.000570224] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0141734 | 61.6784 | 4.13864e-07 | [0.0135353, 0.0148114] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.0041439 | 45.8094 | 1.35817e-06 | [0.00389274, 0.00439505] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 6.9893 | 14.7538 | 0.000122844 | [5.67401, 8.30458] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0.05 | 1.37199 | 0.241982 | [-0.0511832, 0.151183] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000930459 | -7.44822 | 0.00173569 | [-0.0012773, -0.000583615] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midrise | nmse | 5 | 0.0144383 | 78.5632 | 1.57328e-07 | [0.013928, 0.0149485] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.00116445 | 26.1671 | 1.26739e-05 | [0.0010409, 0.00128801] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -2.10637 | -18.1953 | 5.36557e-05 | [-2.42779, -1.78496] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0.025 | 1.63299 | 0.177808 | [-0.0175055, 0.0675055] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.00202034 | -7.61203 | 0.00159869 | [-0.00275724, -0.00128343] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midtread | nmse | 5 | 0.0309541 | 104.187 | 5.08893e-08 | [0.0301292, 0.031779] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.00281023 | 27.7482 | 1.00337e-05 | [0.00252904, 0.00309142] | ok |
| niah | 32768 | 3 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -1.72124 | -8.43377 | 0.00108247 | [-2.28788, -1.15459] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000478985 | -6.74122 | 0.00252364 | [-0.00067626, -0.00028171] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00754063 | 53.417 | 7.35224e-07 | [0.00714869, 0.00793257] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | logit_nmse | 5 | 0.00105945 | 28.4105 | 9.13386e-06 | [0.000955913, 0.00116298] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.77371 | 4.50605 | 0.0107726 | [0.68082, 2.8666] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000477611 | -6.7932 | 0.00245235 | [-0.000672816, -0.000282407] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00751175 | 53.4137 | 7.35403e-07 | [0.00712129, 0.00790221] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00105778 | 28.2298 | 9.36909e-06 | [0.000953743, 0.00116181] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.07626 | 15.4595 | 0.000102176 | [4.1646, 5.98793] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.00010561 | -6.22172 | 0.0033977 | [-0.000152739, -5.84818e-05] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midrise | nmse | 5 | 0.00125183 | 29.0195 | 8.39387e-06 | [0.00113206, 0.0013716] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.000106822 | 15.9321 | 9.07264e-05 | [8.82066e-05, 0.000125438] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -3.14159 | -8.90291 | 0.000879733 | [-4.12132, -2.16186] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.000198671 | -6.52755 | 0.00284499 | [-0.000283175, -0.000114168] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midtread | nmse | 5 | 0.00272529 | 48.5861 | 1.07369e-06 | [0.00256955, 0.00288102] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.000249811 | 21.4849 | 2.7757e-05 | [0.000217529, 0.000282094] | ok |
| niah | 32768 | 4 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -3.16711 | -8.39157 | 0.00110342 | [-4.21499, -2.11924] | ok |
| niah | 32768 | 5 | TQ-GaussianCDF | recall_at_128 | 5 | -0.025 | -1 | 0.373901 | [-0.0944111, 0.0444111] | ok |
| niah | 32768 | 5 | TQ-GaussianCDF | spearman_rho | 5 | -0.000220748 | -6.72373 | 0.00254821 | [-0.000311902, -0.000129594] | ok |
| niah | 32768 | 5 | TQ-GaussianCDF | nmse | 5 | 0.00345327 | 51.1825 | 8.72087e-07 | [0.00326595, 0.0036406] | ok |
| niah | 32768 | 5 | TQ-GaussianCDF | logit_nmse | 5 | 0.000380826 | 25.0795 | 1.50068e-05 | [0.000338666, 0.000422985] | ok |
| niah | 32768 | 5 | TQ-GaussianCDF | codec_us_per_slot | 5 | 0.898074 | 3.08541 | 0.0367378 | [0.089931, 1.70622] | ok |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.025 | -1 | 0.373901 | [-0.0944111, 0.0444111] | ok |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000220038 | -6.76897 | 0.00248527 | [-0.000310292, -0.000129784] | ok |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00344452 | 50.884 | 8.92705e-07 | [0.00325657, 0.00363247] | ok |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000381607 | 24.7673 | 1.57736e-05 | [0.000338828, 0.000424386] | ok |
| niah | 32768 | 5 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.16937 | 19.327 | 4.22459e-05 | [4.42676, 5.91199] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 32768 | 5 | TQ-Uniform-Midrise | spearman_rho | 5 | -6.82846e-06 | -10.1603 | 0.000528423 | [-8.69442e-06, -4.96249e-06] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midrise | nmse | 5 | 3.0747e-05 | 3.32935 | 0.0291233 | [5.10617e-06, 5.63879e-05] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midrise | logit_nmse | 5 | 9.17511e-07 | 0.835827 | 0.450273 | [-2.13027e-06, 3.96529e-06] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -4.05211 | -17.7997 | 5.85346e-05 | [-4.68417, -3.42006] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 32768 | 5 | TQ-Uniform-Midtread | spearman_rho | 5 | -1.83709e-05 | -11.3633 | 0.000342005 | [-2.28595e-05, -1.38823e-05] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midtread | nmse | 5 | 0.000186297 | 19.1841 | 4.35074e-05 | [0.000159335, 0.000213259] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midtread | logit_nmse | 5 | 1.67123e-05 | 15.9522 | 9.02772e-05 | [1.38036e-05, 1.9621e-05] | ok |
| niah | 32768 | 5 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -3.93431 | -15.0247 | 0.000114343 | [-4.66134, -3.20728] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | recall_at_128 | 5 | -0.025 | -1.63299 | 0.177808 | [-0.0675055, 0.0175055] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000924985 | -9.24998 | 0.000759425 | [-0.00120263, -0.000647345] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0141816 | 65.079 | 3.33968e-07 | [0.0135766, 0.0147866] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | logit_nmse | 5 | 0.00415183 | 46.4556 | 1.28428e-06 | [0.0039037, 0.00439997] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.49937 | 15.8762 | 9.19953e-05 | [2.06228, 2.93646] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.025 | -1.63299 | 0.177808 | [-0.0675055, 0.0175055] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000922796 | -9.23189 | 0.000765171 | [-0.00120032, -0.00064527] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0141383 | 65.1052 | 3.33431e-07 | [0.0135354, 0.0147412] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00411472 | 46.1765 | 1.31556e-06 | [0.00386732, 0.00436213] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.93855 | 36.0593 | 3.53067e-06 | [5.4813, 6.3958] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midrise | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000973953 | -9.21242 | 0.000771415 | [-0.00126748, -0.000680423] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midrise | nmse | 5 | 0.0144087 | 72.6008 | 2.15693e-07 | [0.0138577, 0.0149597] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.00117148 | 22.4297 | 2.33953e-05 | [0.00102647, 0.00131649] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -2.18195 | -21.5339 | 2.75068e-05 | [-2.46328, -1.90062] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 131072 | 3 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.00201431 | -8.2198 | 0.00119406 | [-0.0026947, -0.00133393] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midtread | nmse | 5 | 0.0309095 | 105.83 | 4.78034e-08 | [0.0300986, 0.0317204] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.00282872 | 26.5514 | 1.19592e-05 | [0.00253293, 0.00312452] | ok |
| niah | 131072 | 3 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.06723 | -9.141 | 0.000794871 | [-2.69512, -1.43934] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.00047878 | -8.14881 | 0.00123419 | [-0.000641908, -0.000315651] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00751136 | 56.3625 | 5.93309e-07 | [0.00714134, 0.00788137] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | logit_nmse | 5 | 0.00105137 | 28.0891 | 9.55738e-06 | [0.00094745, 0.00115529] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.82278 | 13.3013 | 0.000184663 | [1.4423, 2.20325] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000477134 | -8.16477 | 0.00122503 | [-0.000639384, -0.000314884] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.007483 | 56.2532 | 5.97925e-07 | [0.00711367, 0.00785233] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00104847 | 28.0024 | 9.67575e-06 | [0.00094451, 0.00115242] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.61395 | 22.1762 | 2.44758e-05 | [4.91109, 6.31681] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0 | 0 | 1 | [-0.0548743, 0.0548743] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000113747 | -5.42803 | 0.00558716 | [-0.000171929, -5.55654e-05] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midrise | nmse | 5 | 0.00124848 | 27.4954 | 1.04062e-05 | [0.00112241, 0.00137455] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.000103276 | 18.2788 | 5.26917e-05 | [8.75892e-05, 0.000118963] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -3.05043 | -39.5631 | 2.43861e-06 | [-3.2645, -2.83636] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0 | 0 | 1 | [-0.0548743, 0.0548743] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.000207248 | -6.83808 | 0.00239276 | [-0.000291396, -0.000123099] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midtread | nmse | 5 | 0.0027223 | 51.4523 | 8.53962e-07 | [0.0025754, 0.0028692] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.000249107 | 22.6575 | 2.24744e-05 | [0.000218582, 0.000279633] | ok |
| niah | 131072 | 4 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -3.2386 | -33.8765 | 4.52939e-06 | [-3.50403, -2.97317] | ok |
| niah | 131072 | 5 | TQ-GaussianCDF | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 131072 | 5 | TQ-GaussianCDF | spearman_rho | 5 | -0.000221228 | -7.03318 | 0.00215361 | [-0.000308561, -0.000133895] | ok |
| niah | 131072 | 5 | TQ-GaussianCDF | nmse | 5 | 0.00343131 | 53.925 | 7.0794e-07 | [0.00325464, 0.00360798] | ok |
| niah | 131072 | 5 | TQ-GaussianCDF | logit_nmse | 5 | 0.000379448 | 25.7695 | 1.34705e-05 | [0.000338566, 0.000420331] | ok |
| niah | 131072 | 5 | TQ-GaussianCDF | codec_us_per_slot | 5 | 0.877376 | 5.16715 | 0.00666504 | [0.405939, 1.34881] | ok |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.00022057 | -7.0454 | 0.00213964 | [-0.000307492, -0.000133648] | ok |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00342247 | 53.6328 | 7.23474e-07 | [0.0032453, 0.00359964] | ok |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000379767 | 25.5916 | 1.3847e-05 | [0.000338566, 0.000420968] | ok |
| niah | 131072 | 5 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 4.89806 | 21.0344 | 3.01936e-05 | [4.25154, 5.54458] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midrise | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 131072 | 5 | TQ-Uniform-Midrise | spearman_rho | 5 | -6.71819e-06 | -5.14891 | 0.00674952 | [-1.03408e-05, -3.09555e-06] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midrise | nmse | 5 | 2.74607e-05 | 2.90744 | 0.0437904 | [1.23722e-06, 5.36842e-05] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midrise | logit_nmse | 5 | 5.99694e-07 | 0.942605 | 0.399254 | [-1.16671e-06, 2.36609e-06] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -3.79765 | -45.4814 | 1.39771e-06 | [-4.02948, -3.56582] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midtread | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midtread | spearman_rho | 5 | -1.70171e-05 | -24.1855 | 1.73379e-05 | [-1.89707e-05, -1.50636e-05] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midtread | nmse | 5 | 0.000184503 | 17.2029 | 6.69918e-05 | [0.000154726, 0.000214281] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midtread | logit_nmse | 5 | 1.65311e-05 | 18.6937 | 4.82089e-05 | [1.40759e-05, 1.89864e-05] | ok |
| niah | 131072 | 5 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -3.81168 | -32.8887 | 5.09674e-06 | [-4.13346, -3.4899] | ok |

## Sigmoid-Vasicek parity assessment

Prespecified noninferiority margins: Recall@128=0, rho=0.001. A one-sided 95% lower CI must exceed minus the margin; zero-variance and single-seed outcomes cannot establish statistical noninferiority. No multiplicity correction is applied. A nonsignificant paired t test does not establish parity.

Observed retrieval parity fails: at least one mean Recall@128 or rho loss exceeds its margin.
At least one stratum has a lower mean Recall@128 or rho than Lloyd-Max.

## Unflattened grids: closer to Lloyd-Max or CDF?

For each stratum, distance compares the mean metric with Lloyd-Max and the mean of the two CDF methods. 'Closer' describes absolute distance, not an equivalence test. No statistical observations are pooled across contexts or kappa values.

| Source | Context / kappa | Bits | Uniform regime | NMSE(key) / Lloyd | Closer by NMSE(key) | Closer by NMSE(QK) | Closer by rho |
|---|---|---:|---|---:|---|---|---|
| niah | 8192 | 3 | TQ-Uniform-Midrise | 1.4604 | CDF | Lloyd-Max | CDF |
| niah | 8192 | 3 | TQ-Uniform-Midtread | 1.98809 | CDF | CDF | CDF |
| niah | 8192 | 4 | TQ-Uniform-Midrise | 1.14298 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 8192 | 4 | TQ-Uniform-Midtread | 1.31332 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 8192 | 5 | TQ-Uniform-Midrise | 1.01095 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 8192 | 5 | TQ-Uniform-Midtread | 1.07842 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 32768 | 3 | TQ-Uniform-Midrise | 1.46208 | CDF | Lloyd-Max | CDF |
| niah | 32768 | 3 | TQ-Uniform-Midtread | 1.99064 | CDF | CDF | CDF |
| niah | 32768 | 4 | TQ-Uniform-Midrise | 1.14402 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 32768 | 4 | TQ-Uniform-Midtread | 1.31354 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 32768 | 5 | TQ-Uniform-Midrise | 1.01338 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 32768 | 5 | TQ-Uniform-Midtread | 1.08107 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 131072 | 3 | TQ-Uniform-Midrise | 1.46149 | CDF | Lloyd-Max | CDF |
| niah | 131072 | 3 | TQ-Uniform-Midtread | 1.98999 | CDF | CDF | CDF |
| niah | 131072 | 4 | TQ-Uniform-Midrise | 1.14381 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 131072 | 4 | TQ-Uniform-Midtread | 1.31357 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 131072 | 5 | TQ-Uniform-Midrise | 1.01196 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| niah | 131072 | 5 | TQ-Uniform-Midtread | 1.08033 | Lloyd-Max | Lloyd-Max | Lloyd-Max |

## Does the 5-bit gap disappear?

The following counts use separate context/kappa paired tests at 5 bits. Significant means two-sided p<0.05, without multiplicity adjustment. A nonsignificant difference is not evidence that the gap disappeared. Ratios are ranges of stratum means relative to same-bit Lloyd-Max, not pooled effect estimates.

| Source | Regime | NMSE(key) / Lloyd range | Significant NMSE(key) strata | Significant NMSE(QK) strata | Significant rho strata | Significant recall strata |
|---|---|---|---|---|---|---|
| niah | TQ-GaussianCDF | 2.494–2.503 | 3/3 | 3/3 | 3/3 | 0/3 |
| niah | TQ-Sigmoid-Vasicek | 2.490–2.499 | 3/3 | 3/3 | 3/3 | 0/3 |
| niah | TQ-Uniform-Midrise | 1.011–1.013 | 3/3 | 1/3 | 3/3 | 0/3 |
| niah | TQ-Uniform-Midtread | 1.078–1.081 | 3/3 | 3/3 | 3/3 | 0/3 |
