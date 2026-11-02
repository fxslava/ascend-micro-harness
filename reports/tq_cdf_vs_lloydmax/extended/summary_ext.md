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
| synthetic | 1 | 3 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.981396 | n/a | 0.0339598 | 0.0338498 | 0.983102 | 7.99595 |
| synthetic | 1 | 3 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.970015 | n/a | 0.0566955 | 0.0566518 | 0.972695 | 8 |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.970035 | n/a | 0.0566267 | 0.0565887 | 0.972716 | 8 |
| synthetic | 1 | 3 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.971516 | n/a | 0.0550272 | 0.0549035 | 0.973849 | 7.48091 |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.961521 | n/a | 0.0748758 | 0.0747989 | 0.96495 | 6.58022 |
| synthetic | 1 | 4 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.994883 | n/a | 0.00931563 | 0.00933389 | 0.995413 | 15.2697 |
| synthetic | 1 | 4 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.988043 | n/a | 0.0226153 | 0.0226319 | 0.989163 | 15.9964 |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.988064 | n/a | 0.0225751 | 0.0225948 | 0.989183 | 15.9965 |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.993439 | n/a | 0.011978 | 0.0120318 | 0.994114 | 14.1691 |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.992526 | n/a | 0.0137395 | 0.0137706 | 0.993256 | 13.3699 |
| synthetic | 1 | 5 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.998642 | n/a | 0.00245221 | 0.002455 | 0.998797 | 27.8758 |
| synthetic | 1 | 5 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.995152 | n/a | 0.00917402 | 0.00914907 | 0.995616 | 31.4817 |
| synthetic | 1 | 5 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.995153 | n/a | 0.00917728 | 0.00915433 | 0.995617 | 31.4815 |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.998452 | n/a | 0.0028026 | 0.00280044 | 0.998612 | 25.7189 |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.998352 | n/a | 0.00299415 | 0.00298642 | 0.998518 | 25.0512 |
| synthetic | 8 | 3 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.998669 | n/a | 0.0308678 | 0.00252642 | 0.984484 | 7.99048 |
| synthetic | 8 | 3 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.998103 | n/a | 0.0429809 | 0.00579047 | 0.977801 | 7.99995 |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.998105 | n/a | 0.042943 | 0.00575698 | 0.977814 | 7.99995 |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.998085 | n/a | 0.0435638 | 0.00322727 | 0.978126 | 7.59392 |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.997426 | n/a | 0.0593024 | 0.00438222 | 0.970601 | 6.68235 |
| synthetic | 8 | 4 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.999632 | n/a | 0.00859433 | 0.000657421 | 0.995712 | 14.8396 |
| synthetic | 8 | 4 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.999367 | n/a | 0.0149806 | 0.00132935 | 0.99209 | 15.994 |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.999368 | n/a | 0.0149551 | 0.00132689 | 0.992104 | 15.9939 |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.999576 | n/a | 0.00948743 | 0.000702897 | 0.995099 | 14.6006 |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.999511 | n/a | 0.0108849 | 0.00081262 | 0.994385 | 13.7599 |
| synthetic | 8 | 5 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.9999 | n/a | 0.00227265 | 0.000170539 | 0.998868 | 27.3136 |
| synthetic | 8 | 5 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.99978 | n/a | 0.00515845 | 0.000402356 | 0.997208 | 31.4179 |
| synthetic | 8 | 5 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.99978 | n/a | 0.00514715 | 0.000402454 | 0.997212 | 31.4163 |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.999897 | n/a | 0.00221825 | 0.000164785 | 0.998847 | 26.9795 |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.999891 | n/a | 0.00236939 | 0.00017661 | 0.998769 | 26.2568 |
| synthetic | 64 | 3 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.997868 | n/a | 0.0318173 | 0.00630795 | 0.985272 | 7.34001 |
| synthetic | 64 | 3 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.998149 | n/a | 0.0255961 | 0.0055873 | 0.988072 | 7.81335 |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.998147 | n/a | 0.0256109 | 0.00558634 | 0.98807 | 7.81328 |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.998527 | n/a | 0.023383 | 0.00369286 | 0.988472 | 7.84351 |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.997622 | n/a | 0.0314212 | 0.00603106 | 0.984733 | 6.89971 |
| synthetic | 64 | 4 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.999779 | n/a | 0.00831476 | 0.000653689 | 0.995964 | 12.361 |
| synthetic | 64 | 4 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.999674 | n/a | 0.00920502 | 0.00152496 | 0.995993 | 14.7498 |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.999674 | n/a | 0.00920174 | 0.0015191 | 0.995992 | 14.7494 |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.999895 | n/a | 0.00511942 | 0.000238711 | 0.997374 | 14.7919 |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.999869 | n/a | 0.00589311 | 0.000301608 | 0.996985 | 13.9436 |
| synthetic | 64 | 5 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.999966 | n/a | 0.00215306 | 7.61499e-05 | 0.998936 | 21.7035 |
| synthetic | 64 | 5 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.999939 | n/a | 0.00320215 | 0.000274874 | 0.998519 | 26.3671 |
| synthetic | 64 | 5 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.99994 | n/a | 0.00317576 | 0.000269009 | 0.998529 | 26.3639 |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | n/a | n/a | n/a | n/a | 0.99998 | n/a | 0.00118012 | 3.76488e-05 | 0.999387 | 26.8672 |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | n/a | n/a | n/a | n/a | 0.999979 | n/a | 0.00125846 | 4.03308e-05 | 0.999346 | 26.1495 |
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
| synthetic | 1 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.174194 | 21.2639 | 7.95143 | 3.125 |
| synthetic | 1 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.196718 | 24.0134 | 10.6617 | 3.125 |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.224645 | 27.4225 | 14.0302 | 3.125 |
| synthetic | 1 | 3 | TQ-Uniform-Midrise | 0.436142 | 11.8961 | 0.156235 | 19.0716 | 5.73171 | 3.125 |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | 0.436142 | 13.7418 | 0.154563 | 18.8675 | 5.53019 | 3.125 |
| synthetic | 1 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.187099 | 22.8392 | 9.48351 | 4.125 |
| synthetic | 1 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.195191 | 23.827 | 10.5346 | 4.125 |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.228345 | 27.8742 | 14.5214 | 4.125 |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | 0.436142 | 5.6 | 0.158888 | 19.3955 | 6.07648 | 4.125 |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | 0.436142 | 5.98984 | 0.159457 | 19.4649 | 6.11758 | 4.125 |
| synthetic | 1 | 5 | TQ-LloydMax | 0.436142 | 2.73921 | 0.197251 | 24.0784 | 10.7668 | 5.125 |
| synthetic | 1 | 5 | TQ-GaussianCDF | 0.436142 | 4.70046 | 0.201146 | 24.554 | 11.2147 | 5.125 |
| synthetic | 1 | 5 | TQ-Sigmoid-Vasicek | 0.436142 | 4.69729 | 0.23366 | 28.5229 | 15.0995 | 5.125 |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | 0.436142 | 2.71605 | 0.16161 | 19.7278 | 6.36697 | 5.125 |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | 0.436142 | 2.8164 | 0.164324 | 20.0591 | 6.69702 | 5.125 |
| synthetic | 8 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.18066 | 22.0532 | 8.05104 | 3.125 |
| synthetic | 8 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.199184 | 24.3144 | 10.3106 | 3.125 |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.238867 | 29.1586 | 15.1086 | 3.125 |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | 0.436142 | 11.8961 | 0.159463 | 19.4657 | 5.46846 | 3.125 |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | 0.436142 | 13.7418 | 0.159313 | 19.4474 | 5.47546 | 3.125 |
| synthetic | 8 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.191111 | 23.329 | 9.34474 | 4.125 |
| synthetic | 8 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.204494 | 24.9626 | 10.9684 | 4.125 |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.233385 | 28.4894 | 14.4835 | 4.125 |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | 0.436142 | 5.6 | 0.167019 | 20.3881 | 6.41906 | 4.125 |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | 0.436142 | 5.98984 | 0.168556 | 20.5757 | 6.59139 | 4.125 |
| synthetic | 8 | 5 | TQ-LloydMax | 0.436142 | 2.73921 | 0.20411 | 24.9157 | 10.847 | 5.125 |
| synthetic | 8 | 5 | TQ-GaussianCDF | 0.436142 | 4.70046 | 0.204778 | 24.9973 | 11.0208 | 5.125 |
| synthetic | 8 | 5 | TQ-Sigmoid-Vasicek | 0.436142 | 4.69729 | 0.240295 | 29.3328 | 15.2871 | 5.125 |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | 0.436142 | 2.71605 | 0.169041 | 20.6349 | 6.63941 | 5.125 |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | 0.436142 | 2.8164 | 0.168921 | 20.6202 | 6.57927 | 5.125 |
| synthetic | 64 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.175983 | 21.4823 | 7.57714 | 3.125 |
| synthetic | 64 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.199486 | 24.3513 | 10.4729 | 3.125 |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.23486 | 28.6695 | 14.7472 | 3.125 |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | 0.436142 | 11.8961 | 0.16442 | 20.0708 | 6.11783 | 3.125 |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | 0.436142 | 13.7418 | 0.162257 | 19.8068 | 5.91498 | 3.125 |
| synthetic | 64 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.181318 | 22.1336 | 8.24026 | 4.125 |
| synthetic | 64 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.203127 | 24.7958 | 10.9214 | 4.125 |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.240713 | 29.3839 | 15.4257 | 4.125 |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | 0.436142 | 5.6 | 0.165534 | 20.2068 | 6.32031 | 4.125 |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | 0.436142 | 5.98984 | 0.1621 | 19.7876 | 5.889 | 4.125 |
| synthetic | 64 | 5 | TQ-LloydMax | 0.436142 | 2.73921 | 0.193854 | 23.6639 | 9.80071 | 5.125 |
| synthetic | 64 | 5 | TQ-GaussianCDF | 0.436142 | 4.70046 | 0.203109 | 24.7936 | 10.9134 | 5.125 |
| synthetic | 64 | 5 | TQ-Sigmoid-Vasicek | 0.436142 | 4.69729 | 0.233745 | 28.5334 | 14.6115 | 5.125 |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | 0.436142 | 2.71605 | 0.167913 | 20.4972 | 6.58781 | 5.125 |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | 0.436142 | 2.8164 | 0.174203 | 21.265 | 7.35796 | 5.125 |
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
| synthetic | 1 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.0113809 | -70.6237 | 2.40862e-07 | [-0.0118284, -0.0109335] | ok |
| synthetic | 1 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0227357 | 349.601 | 4.0164e-10 | [0.0225551, 0.0229162] | ok |
| synthetic | 1 | 3 | TQ-GaussianCDF | logit_nmse | 5 | 0.022802 | 234.847 | 1.97223e-09 | [0.0225324, 0.0230715] | ok |
| synthetic | 1 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.7103 | 9.4169 | 0.000708852 | [1.91121, 3.5094] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.011361 | -72.1974 | 2.20552e-07 | [-0.0117979, -0.0109241] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0226668 | 349.92 | 4.00178e-10 | [0.022487, 0.0228467] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.0227389 | 230.342 | 2.13111e-09 | [0.0224649, 0.023013] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 6.07872 | 18.6072 | 4.91036e-05 | [5.17169, 6.98575] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.00988008 | -135.923 | 1.75718e-08 | [-0.0100819, -0.00967826] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midrise | nmse | 5 | 0.0210674 | 282.055 | 9.4794e-10 | [0.02086, 0.0212748] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.0210537 | 164.942 | 8.10436e-09 | [0.0206993, 0.0214081] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -2.21972 | -7.83835 | 0.00143067 | [-3.00597, -1.43346] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.0198752 | -100.097 | 5.97274e-08 | [-0.0204265, -0.019324] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | nmse | 5 | 0.040916 | 553.825 | 6.37752e-11 | [0.0407109, 0.0411211] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.0409491 | 152.068 | 1.12169e-08 | [0.0402015, 0.0416968] | ok |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.42124 | -8.03003 | 0.00130516 | [-3.2584, -1.58407] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.00684004 | -137.543 | 1.6759e-08 | [-0.00697812, -0.00670197] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | nmse | 5 | 0.0132997 | 328.664 | 5.14179e-10 | [0.0131874, 0.0134121] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | logit_nmse | 5 | 0.0132981 | 282.92 | 9.36399e-10 | [0.0131676, 0.0134286] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.05111 | 2.77055 | 0.0503029 | [-0.00223837, 2.10446] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.00681911 | -139.869 | 1.56718e-08 | [-0.00695447, -0.00668375] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0132595 | 330.987 | 4.99895e-10 | [0.0131483, 0.0133707] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.013261 | 297.98 | 7.60972e-10 | [0.0131374, 0.0133845] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.03786 | 12.3582 | 0.000246379 | [3.90603, 6.16968] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.00144406 | -68.0931 | 2.78685e-07 | [-0.00150294, -0.00138518] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | nmse | 5 | 0.00266235 | 427.354 | 1.79881e-10 | [0.00264505, 0.00267965] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.00269791 | 103.066 | 5.31399e-08 | [0.00262524, 0.00277059] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -3.40703 | -7.5164 | 0.00167697 | [-4.66553, -2.14852] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.00235764 | -68.7015 | 2.6895e-07 | [-0.00245292, -0.00226236] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | nmse | 5 | 0.00442383 | 311.963 | 6.33442e-10 | [0.00438446, 0.0044632] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.00443672 | 217.025 | 2.70428e-09 | [0.00437996, 0.00449348] | ok |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -3.36594 | -8.09324 | 0.00126678 | [-4.52065, -2.21123] | ok |
| synthetic | 1 | 5 | TQ-GaussianCDF | spearman_rho | 5 | -0.00348938 | -84.9866 | 1.14908e-07 | [-0.00360338, -0.00337539] | ok |
| synthetic | 1 | 5 | TQ-GaussianCDF | nmse | 5 | 0.0067218 | 233.835 | 2.00659e-09 | [0.00664199, 0.00680161] | ok |
| synthetic | 1 | 5 | TQ-GaussianCDF | logit_nmse | 5 | 0.00669408 | 272.219 | 1.09255e-09 | [0.0066258, 0.00676235] | ok |
| synthetic | 1 | 5 | TQ-GaussianCDF | codec_us_per_slot | 5 | 0.447961 | 2.50399 | 0.0664829 | [-0.0487428, 0.944666] | ok |
| synthetic | 1 | 5 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.00348837 | -85.8927 | 1.10137e-07 | [-0.00360113, -0.00337561] | ok |
| synthetic | 1 | 5 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00672507 | 231.88 | 2.07512e-09 | [0.00664454, 0.00680559] | ok |
| synthetic | 1 | 5 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00669933 | 260.009 | 1.31267e-09 | [0.0066278, 0.00677087] | ok |
| synthetic | 1 | 5 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 4.33268 | 10.7891 | 0.000418546 | [3.21771, 5.44765] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000189836 | -20.9477 | 3.06928e-05 | [-0.000214998, -0.000164675] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | nmse | 5 | 0.000350385 | 162.739 | 8.55212e-09 | [0.000344407, 0.000356363] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.000345437 | 44.1406 | 1.57512e-06 | [0.000323709, 0.000367165] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -4.39981 | -12.6141 | 0.000227373 | [-5.36824, -3.43139] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.000290039 | -53.2874 | 7.42396e-07 | [-0.000305151, -0.000274927] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | nmse | 5 | 0.000541935 | 231.927 | 2.07345e-09 | [0.000535448, 0.000548423] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.000531422 | 100.79 | 5.81029e-08 | [0.000516783, 0.000546061] | ok |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -4.06976 | -19.2231 | 4.31582e-05 | [-4.65756, -3.48195] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000565927 | -23.2718 | 2.02071e-05 | [-0.000633446, -0.000498409] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0121131 | 335.368 | 4.74283e-10 | [0.0120128, 0.0122134] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | logit_nmse | 5 | 0.00326404 | 112.77 | 3.70806e-08 | [0.00318368, 0.00334441] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.25955 | 7.46639 | 0.0017198 | [1.41931, 3.09978] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000564054 | -22.878 | 2.16257e-05 | [-0.000632507, -0.000495601] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0120752 | 335.292 | 4.74713e-10 | [0.0119752, 0.0121752] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.00323056 | 114.002 | 3.55047e-08 | [0.00315188, 0.00330924] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.05753 | 11.8535 | 0.000290021 | [5.40445, 8.71061] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | spearman_rho | 5 | -0.000584355 | -18.6236 | 4.89318e-05 | [-0.000671472, -0.000497238] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | nmse | 5 | 0.012696 | 399.704 | 2.35061e-10 | [0.0126078, 0.0127842] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | logit_nmse | 5 | 0.000700842 | 41.5764 | 2.00027e-06 | [0.000654041, 0.000747644] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -2.58258 | -8.3042 | 0.00114842 | [-3.44604, -1.71911] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.00124279 | -15.4485 | 0.000102464 | [-0.00146615, -0.00101943] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | nmse | 5 | 0.0284346 | 412.334 | 2.07557e-10 | [0.0282431, 0.0286261] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.0018558 | 42.5725 | 1.81986e-06 | [0.00173477, 0.00197683] | ok |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.57558 | -15.2876 | 0.000106783 | [-3.04334, -2.10782] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000265233 | -11.5652 | 0.000319295 | [-0.000328907, -0.000201559] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00638626 | 276.331 | 1.02895e-09 | [0.00632209, 0.00645042] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | logit_nmse | 5 | 0.000671925 | 99.5628 | 6.10199e-08 | [0.000653187, 0.000690662] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.62362 | 3.7687 | 0.0196265 | [0.427481, 2.81977] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000263817 | -11.5386 | 0.000322179 | [-0.000327297, -0.000200337] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0063608 | 273.82 | 1.06721e-09 | [0.00629631, 0.0064253] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000669473 | 96.9218 | 6.79449e-08 | [0.000650295, 0.00068865] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.13876 | 11.0808 | 0.000377263 | [3.85118, 6.42634] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | spearman_rho | 5 | -5.60693e-05 | -14.1786 | 0.000143663 | [-6.70488e-05, -4.50899e-05] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | nmse | 5 | 0.000893101 | 66.6172 | 3.04196e-07 | [0.000855879, 0.000930323] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | logit_nmse | 5 | 4.54752e-05 | 32.5108 | 5.33709e-06 | [4.15916e-05, 4.93588e-05] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -2.92568 | -6.81163 | 0.00242766 | [-4.1182, -1.73316] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.000121289 | -18.7977 | 4.71607e-05 | [-0.000139203, -0.000103374] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | nmse | 5 | 0.00229062 | 262.653 | 1.26061e-09 | [0.00226641, 0.00231483] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | logit_nmse | 5 | 0.000155199 | 44.5712 | 1.51523e-06 | [0.000145531, 0.000164867] | ok |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.75335 | -11.6028 | 0.000315273 | [-3.4122, -2.0945] | ok |
| synthetic | 8 | 5 | TQ-GaussianCDF | spearman_rho | 5 | -0.000120596 | -12.4855 | 0.000236687 | [-0.000147413, -9.37786e-05] | ok |
| synthetic | 8 | 5 | TQ-GaussianCDF | nmse | 5 | 0.0028858 | 298.906 | 7.51586e-10 | [0.002859, 0.00291261] | ok |
| synthetic | 8 | 5 | TQ-GaussianCDF | logit_nmse | 5 | 0.000231817 | 42.3023 | 1.86672e-06 | [0.000216602, 0.000247032] | ok |
| synthetic | 8 | 5 | TQ-GaussianCDF | codec_us_per_slot | 5 | 0.173784 | 0.76447 | 0.487196 | [-0.457375, 0.804943] | ok |
| synthetic | 8 | 5 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000120321 | -12.4872 | 0.000236563 | [-0.000147073, -9.35684e-05] | ok |
| synthetic | 8 | 5 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0028745 | 287.75 | 8.7509e-10 | [0.00284677, 0.00290224] | ok |
| synthetic | 8 | 5 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000231915 | 42.8293 | 1.77668e-06 | [0.000216881, 0.000246949] | ok |
| synthetic | 8 | 5 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 4.44011 | 12.4884 | 0.000236475 | [3.45298, 5.42725] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | spearman_rho | 5 | -2.93255e-06 | -6.20907 | 0.00342329 | [-4.24387e-06, -1.62123e-06] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | nmse | 5 | -5.44006e-05 | -21.8452 | 2.59828e-05 | [-6.13147e-05, -4.74864e-05] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | logit_nmse | 5 | -5.75421e-06 | -6.39431 | 0.00307104 | [-8.25272e-06, -3.2557e-06] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -4.20761 | -36.1904 | 3.47993e-06 | [-4.53041, -3.88482] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | spearman_rho | 5 | -9.04724e-06 | -16.1752 | 8.54608e-05 | [-1.06002e-05, -7.4943e-06] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | nmse | 5 | 9.67437e-05 | 34.82 | 4.05929e-06 | [8.90297e-05, 0.000104458] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | logit_nmse | 5 | 6.0709e-06 | 4.30164 | 0.0126292 | [2.15251e-06, 9.9893e-06] | ok |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -4.26775 | -14.9516 | 0.000116561 | [-5.06025, -3.47525] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | spearman_rho | 5 | 0.00028092 | 7.40087 | 0.00177798 | [0.000175532, 0.000386307] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | nmse | 5 | -0.0062212 | -64.9141 | 3.37372e-07 | [-0.00648729, -0.00595511] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | logit_nmse | 5 | -0.00072065 | -15.7548 | 9.48246e-05 | [-0.000847648, -0.000593651] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.89579 | 9.15924 | 0.000788799 | [2.01799, 3.77359] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | 0.000278663 | 7.32039 | 0.0018528 | [0.000172973, 0.000384353] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | -0.00620636 | -64.6182 | 3.4359e-07 | [-0.00647303, -0.00593969] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | -0.000721614 | -15.6619 | 9.7065e-05 | [-0.000849537, -0.000593691] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.1701 | 23.1179 | 2.07474e-05 | [6.30897, 8.03122] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | spearman_rho | 5 | 0.000659087 | 29.1991 | 8.19007e-06 | [0.000596416, 0.000721757] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | nmse | 5 | -0.00843424 | -138.661 | 1.6225e-08 | [-0.00860312, -0.00826536] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | logit_nmse | 5 | -0.00261509 | -63.4909 | 3.68627e-07 | [-0.00272944, -0.00250073] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -1.45931 | -2.8954 | 0.0443218 | [-2.85867, -0.059956] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | spearman_rho | 5 | -0.000246176 | -11.2241 | 0.000358849 | [-0.000307072, -0.000185281] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | nmse | 5 | -0.000396125 | -5.23454 | 0.00636391 | [-0.000606233, -0.000186017] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | logit_nmse | 5 | -0.000276887 | -5.46227 | 0.00546206 | [-0.000417627, -0.000136147] | ok |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -1.66217 | -4.28122 | 0.0128353 | [-2.74011, -0.584221] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000104854 | -38.9773 | 2.58821e-06 | [-0.000112322, -9.73846e-05] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | nmse | 5 | 0.000890259 | 96.738 | 6.84627e-08 | [0.000864708, 0.00091581] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | logit_nmse | 5 | 0.000871267 | 76.5871 | 1.74195e-07 | [0.000839682, 0.000902853] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.68113 | 12.3108 | 0.000250116 | [2.07645, 3.2858] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000104417 | -35.8184 | 3.6264e-06 | [-0.000112511, -9.63231e-05] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.000886979 | 94.5321 | 7.50775e-08 | [0.000860928, 0.000913029] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000865407 | 77.4975 | 1.66157e-07 | [0.000834402, 0.000896411] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.18541 | 17.5649 | 6.16943e-05 | [6.04962, 8.32119] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | spearman_rho | 5 | 0.000116058 | 68.2913 | 2.75466e-07 | [0.000111339, 0.000120776] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | nmse | 5 | -0.00319534 | -689.415 | 2.65596e-11 | [-0.0032082, -0.00318247] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | logit_nmse | 5 | -0.000414977 | -105.305 | 4.8763e-08 | [-0.000425918, -0.000404036] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -1.91995 | -4.61596 | 0.00991112 | [-3.07478, -0.765124] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | spearman_rho | 5 | 9.08166e-05 | 37.1191 | 3.14532e-06 | [8.40237e-05, 9.76096e-05] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | nmse | 5 | -0.00242165 | -511.482 | 8.76638e-11 | [-0.00243479, -0.0024085] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | logit_nmse | 5 | -0.00035208 | -135.811 | 1.76299e-08 | [-0.000359278, -0.000344883] | ok |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.35126 | -15.1542 | 0.000110538 | [-2.78204, -1.92048] | ok |
| synthetic | 64 | 5 | TQ-GaussianCDF | spearman_rho | 5 | -2.76007e-05 | -63.0249 | 3.79643e-07 | [-2.88166e-05, -2.63848e-05] | ok |
| synthetic | 64 | 5 | TQ-GaussianCDF | nmse | 5 | 0.0010491 | 128.722 | 2.18459e-08 | [0.00102647, 0.00107173] | ok |
| synthetic | 64 | 5 | TQ-GaussianCDF | logit_nmse | 5 | 0.000198724 | 86.409 | 1.0753e-07 | [0.000192339, 0.00020511] | ok |
| synthetic | 64 | 5 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.11267 | 3.64189 | 0.0219279 | [0.264412, 1.96093] | ok |
| synthetic | 64 | 5 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -2.65852e-05 | -57.9051 | 5.32625e-07 | [-2.78599e-05, -2.53105e-05] | ok |
| synthetic | 64 | 5 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00102271 | 124.835 | 2.46957e-08 | [0.00099996, 0.00104545] | ok |
| synthetic | 64 | 5 | TQ-Sigmoid-Vasicek | logit_nmse | 5 | 0.000192859 | 74.1387 | 1.98356e-07 | [0.000185636, 0.000200081] | ok |
| synthetic | 64 | 5 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 4.81076 | 48.3697 | 1.093e-06 | [4.53462, 5.0869] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | spearman_rho | 5 | 1.36845e-05 | 72.1543 | 2.21078e-07 | [1.31579e-05, 1.42111e-05] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | nmse | 5 | -0.000972934 | -461.386 | 1.32398e-10 | [-0.000978789, -0.00096708] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | logit_nmse | 5 | -3.85011e-05 | -101.118 | 5.73524e-08 | [-3.95583e-05, -3.7444e-05] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | codec_us_per_slot | 5 | -3.2129 | -9.44842 | 0.000699777 | [-4.15702, -2.26878] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | spearman_rho | 5 | 1.24559e-05 | 100.582 | 5.85841e-08 | [1.21121e-05, 1.27997e-05] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | nmse | 5 | -0.000894594 | -516.272 | 8.44548e-11 | [-0.000899405, -0.000889783] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | logit_nmse | 5 | -3.58191e-05 | -108.019 | 4.40459e-08 | [-3.67398e-05, -3.48984e-05] | ok |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | codec_us_per_slot | 5 | -2.44275 | -4.8721 | 0.00820723 | [-3.8348, -1.05071] | ok |
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
| synthetic | 1 | 3 | TQ-Uniform-Midrise | 1.62036 | CDF | CDF | CDF |
| synthetic | 1 | 3 | TQ-Uniform-Midtread | 2.20484 | CDF | CDF | CDF |
| synthetic | 1 | 4 | TQ-Uniform-Midrise | 1.28579 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 1 | 4 | TQ-Uniform-Midtread | 1.47488 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 1 | 5 | TQ-Uniform-Midrise | 1.14289 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 1 | 5 | TQ-Uniform-Midtread | 1.221 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 8 | 3 | TQ-Uniform-Midrise | 1.4113 | CDF | Lloyd-Max | CDF |
| synthetic | 8 | 3 | TQ-Uniform-Midtread | 1.92117 | CDF | CDF | CDF |
| synthetic | 8 | 4 | TQ-Uniform-Midrise | 1.10392 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 8 | 4 | TQ-Uniform-Midtread | 1.26653 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 8 | 5 | TQ-Uniform-Midrise | 0.976063 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 8 | 5 | TQ-Uniform-Midtread | 1.04257 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 64 | 3 | TQ-Uniform-Midrise | 0.734916 | CDF | CDF | CDF |
| synthetic | 64 | 3 | TQ-Uniform-Midtread | 0.98755 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 64 | 4 | TQ-Uniform-Midrise | 0.615703 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 64 | 4 | TQ-Uniform-Midtread | 0.708753 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 64 | 5 | TQ-Uniform-Midrise | 0.548115 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
| synthetic | 64 | 5 | TQ-Uniform-Midtread | 0.5845 | Lloyd-Max | Lloyd-Max | Lloyd-Max |
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
| synthetic | TQ-GaussianCDF | 1.487–3.741 | 3/3 | 3/3 | 3/3 | n/a |
| synthetic | TQ-Sigmoid-Vasicek | 1.475–3.742 | 3/3 | 3/3 | 3/3 | n/a |
| synthetic | TQ-Uniform-Midrise | 0.548–1.143 | 3/3 | 3/3 | 3/3 | n/a |
| synthetic | TQ-Uniform-Midtread | 0.585–1.221 | 3/3 | 3/3 | 3/3 | n/a |

## Cross-bit parity: plain uniform versus Lloyd-Max / inverse CDF

Comparisons: **5-bit uniform vs 4-bit reference**, and **4-bit uniform vs 3-bit reference**. 'Plain' retains the common D H D rotation and one FP16 absmax-derived step per vector; it omits CDF flattening and Lloyd-Max codebooks. Both midrise and midtread are tested. The three references are Lloyd-Max, exact Gaussian-CDF, and sigmoid-CDF. The same five seeds are paired within each context or synthetic kappa stratum.

Full paired differences, two-sided t tests and 95% CIs for all metrics are in cross_bit.csv. Practical equivalence uses paired TOST at alpha=.05 (a 90% CI inside the band): Recall/FPR +/-1/16, rho +/-0.001, and NMSE ratios within [1/1.10,1.10] (tested in log space). These are explicit working tolerances, not a claim that any loss is acceptable to every application. Strict zero-loss noninferiority is recorded separately. No multiplicity correction; zero-variance or single-seed tests remain inconclusive. A better method can fail equivalence because it lies outside the band in the favorable direction.

Table labels: equiv=TOST equivalence within the working band; better/worse=one-sided 95% zero-loss test in the indicated direction; unresolved=neither conclusion supported. These labels do not replace the paired 95% difference CIs.

## NIAH synthesis

Ranges below span the three contexts and both uniform grids. They summarize separate strata; confidence intervals and tests remain paired within each stratum. CDF includes exact Gaussian and sigmoid variants.

| Comparison | Reference family | NMSE(key) ratio | NMSE(QK) ratio | Rho delta | Recall delta (percentage points) |
|---|---|---|---|---|---|
| uniform 5 vs reference 4 bits | Lloyd-Max | 0.267723 to 0.285812 | 0.261279 to 0.283442 | 0.000388912 to 0.000419644 | 0 to 1.25 |
| uniform 5 vs reference 4 bits | CDF | 0.143438 to 0.153504 | 0.117937 to 0.12835 | 0.000866046 to 0.00089962 | -1.25 to 2.5 |
| uniform 4 vs reference 3 bits | Lloyd-Max | 0.317632 to 0.365391 | 0.290357 to 0.334885 | 0.0012918 to 0.00141564 | -1.25 to 5 |
| uniform 4 vs reference 3 bits | CDF | 0.218291 to 0.25141 | 0.130464 to 0.151735 | 0.0022146 to 0.00236055 | 1.25 to 3.75 |

The extra-bit uniform method has a favorable two-sided 95% difference CI in 108/108 fidelity comparisons (key NMSE, logit NMSE, rho). Mean recall is lower in 4/36 comparisons; 0/36 recall difference tests have p<0.05. This supports the fidelity advantage on this grid, while strict loss-free retrieval parity is not established. Absence of a significant recall difference is not proof of parity.

## niah: paired cross-bit results

Ratio <1 means less reconstruction/projection error for uniform. Positive rho/recall delta favors uniform. NMSE(key) and NMSE(QK) are distinct measures.

| Context / kappa | Comparison | Uniform grid | Reference | Recall (uniform / ref) | Rho delta [95% CI] | NMSE(key) ratio | NMSE(QK) ratio | TOST: recall / rho / key / QK |
|---|---|---|---|---|---|---:|---:|---|
| 8192 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-LloydMax | 0.5625 / 0.5625 | 0.000419644 [0.000257374, 0.000581913] | 0.267775 | 0.26153 | unresolved / equiv / better / better |
| 8192 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | 0.5625 / 0.5625 | 0.00089962 [0.000555202, 0.00124404] | 0.143438 | 0.117937 | unresolved / better / better / better |
| 8192 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | 0.5625 / 0.5625 | 0.000899556 [0.000554175, 0.00124494] | 0.14368 | 0.118098 | unresolved / better / better / better |
| 8192 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-LloydMax | 0.5625 / 0.5625 | 0.0004065 [0.000249616, 0.000563384] | 0.285645 | 0.283442 | unresolved / equiv / better / better |
| 8192 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | 0.5625 / 0.5625 | 0.000886476 [0.000547489, 0.00122546] | 0.15301 | 0.127819 | unresolved / better / better / better |
| 8192 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | 0.5625 / 0.5625 | 0.000886412 [0.000546475, 0.00122635] | 0.153269 | 0.127993 | unresolved / better / better / better |
| 8192 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-LloydMax | 0.5625 / 0.55 | 0.00141564 [0.000874916, 0.00195637] | 0.317632 | 0.290357 | equiv / better / better / better |
| 8192 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | 0.5625 / 0.5375 | 0.00236052 [0.001497, 0.00322403] | 0.218291 | 0.130464 | equiv / better / better / better |
| 8192 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | 0.5625 / 0.5375 | 0.00236055 [0.00149814, 0.00322296] | 0.2185 | 0.131153 | equiv / better / better / better |
| 8192 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-LloydMax | 0.575 / 0.55 | 0.00132835 [0.000796345, 0.00186036] | 0.36497 | 0.330505 | equiv / better / better / better |
| 8192 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | 0.575 / 0.5375 | 0.00227322 [0.00141831, 0.00312814] | 0.250824 | 0.148504 | better / better / better / better |
| 8192 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | 0.575 / 0.5375 | 0.00227326 [0.00141947, 0.00312705] | 0.251063 | 0.149288 | better / better / better / better |
| 32768 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-LloydMax | 0.4125 / 0.4 | 0.00040817 [0.000255461, 0.000560879] | 0.267902 | 0.262158 | equiv / equiv / better / better |
| 32768 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | 0.4125 / 0.3875 | 0.000887156 [0.000537878, 0.00123643] | 0.143451 | 0.119778 | unresolved / better / better / better |
| 32768 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | 0.4125 / 0.3875 | 0.000885782 [0.000538541, 0.00123302] | 0.143707 | 0.119881 | unresolved / better / better / better |
| 32768 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-LloydMax | 0.4125 / 0.4 | 0.000396628 [0.0002467, 0.000546556] | 0.285798 | 0.279879 | equiv / equiv / better / better |
| 32768 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | 0.4125 / 0.3875 | 0.000875613 [0.000529127, 0.0012221] | 0.153034 | 0.127875 | unresolved / better / better / better |
| 32768 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | 0.4125 / 0.3875 | 0.000874239 [0.00052979, 0.00121869] | 0.153307 | 0.127985 | unresolved / better / better / better |
| 32768 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-LloydMax | 0.4125 / 0.3625 | 0.00139846 [0.00086978, 0.00192714] | 0.318235 | 0.292732 | unresolved / better / better / better |
| 32768 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | 0.4125 / 0.4 | 0.00233319 [0.00144005, 0.00322634] | 0.218721 | 0.131485 | equiv / better / better / better |
| 32768 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | 0.4125 / 0.4 | 0.00233183 [0.00144119, 0.00322247] | 0.218929 | 0.132137 | equiv / better / better / better |
| 32768 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-LloydMax | 0.4125 / 0.3625 | 0.0013054 [0.000814179, 0.00179662] | 0.365391 | 0.334669 | unresolved / better / better / better |
| 32768 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | 0.4125 / 0.4 | 0.00224013 [0.00138446, 0.0030958] | 0.251131 | 0.150322 | equiv / better / better / better |
| 32768 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | 0.4125 / 0.4 | 0.00223877 [0.00138561, 0.00309193] | 0.25137 | 0.151067 | equiv / better / better / better |
| 131072 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-LloydMax | 0.1875 / 0.1875 | 0.000399211 [0.00025449, 0.000543932] | 0.267723 | 0.261279 | unresolved / equiv / better / better |
| 131072 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | 0.1875 / 0.2 | 0.000877991 [0.000571579, 0.0011844] | 0.143537 | 0.119962 | equiv / better / better / better |
| 131072 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | 0.1875 / 0.2 | 0.000876345 [0.000570838, 0.00118185] | 0.143789 | 0.120142 | equiv / better / better / better |
| 131072 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-LloydMax | 0.2 / 0.1875 | 0.000388912 [0.000244034, 0.000533791] | 0.285812 | 0.279129 | equiv / equiv / better / better |
| 131072 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | 0.2 / 0.2 | 0.000867692 [0.000560864, 0.00117452] | 0.153235 | 0.128158 | unresolved / better / better / better |
| 131072 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | 0.2 / 0.2 | 0.000866046 [0.000560121, 0.00117197] | 0.153504 | 0.12835 | unresolved / better / better / better |
| 131072 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-LloydMax | 0.1875 / 0.2 | 0.0013853 [0.000853963, 0.00191664] | 0.318052 | 0.292106 | equiv / better / better / better |
| 131072 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | 0.1875 / 0.175 | 0.00231029 [0.00150347, 0.0031171] | 0.21871 | 0.131703 | equiv / better / better / better |
| 131072 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | 0.1875 / 0.175 | 0.0023081 [0.0015012, 0.003115] | 0.218919 | 0.132352 | equiv / better / better / better |
| 131072 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-LloydMax | 0.1875 / 0.2 | 0.0012918 [0.000787958, 0.00179565] | 0.365257 | 0.334885 | equiv / better / better / better |
| 131072 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | 0.1875 / 0.175 | 0.00221679 [0.0014374, 0.00299617] | 0.251171 | 0.15099 | equiv / better / better / better |
| 131072 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | 0.1875 / 0.175 | 0.0022146 [0.00143514, 0.00299406] | 0.25141 | 0.151735 | equiv / better / better / better |
## synthetic: paired cross-bit results

Ratio <1 means less reconstruction/projection error for uniform. Positive rho/recall delta favors uniform. NMSE(key) and NMSE(QK) are distinct measures.

| Context / kappa | Comparison | Uniform grid | Reference | Recall (uniform / ref) | Rho delta [95% CI] | NMSE(key) ratio | NMSE(QK) ratio | TOST: recall / rho / key / QK |
|---|---|---|---|---|---|---:|---:|---|
| 1.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-LloydMax | n/a | 0.00356873 [0.00350308, 0.00363438] | 0.300849 | 0.300029 | n/a / better / better / better |
| 1.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | n/a | 0.0104088 [0.0102251, 0.0105925] | 0.123925 | 0.123738 | n/a / better / better / better |
| 1.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | n/a | 0.0103878 [0.0102049, 0.0105708] | 0.124145 | 0.123941 | n/a / better / better / better |
| 1.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-LloydMax | n/a | 0.00346853 [0.00339494, 0.00354212] | 0.321411 | 0.319954 | n/a / better / better / better |
| 1.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | n/a | 0.0103086 [0.0101172, 0.0104999] | 0.132395 | 0.131956 | n/a / better / better / better |
| 1.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | n/a | 0.0102876 [0.0100972, 0.010478] | 0.13263 | 0.132173 | n/a / better / better / better |
| 1.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-LloydMax | n/a | 0.0120429 [0.0118456, 0.0122403] | 0.35271 | 0.355447 | n/a / better / better / better |
| 1.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | n/a | 0.0234239 [0.0230135, 0.0238343] | 0.211269 | 0.212382 | n/a / better / better / better |
| 1.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | n/a | 0.023404 [0.0229993, 0.0238086] | 0.211525 | 0.212618 | n/a / better / better / better |
| 1.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-LloydMax | n/a | 0.0111294 [0.0109688, 0.0112899] | 0.40458 | 0.406815 | n/a / better / better / better |
| 1.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | n/a | 0.0225103 [0.0221513, 0.0228693] | 0.242338 | 0.243075 | n/a / better / better / better |
| 1.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | n/a | 0.0224904 [0.0221395, 0.0228413] | 0.242632 | 0.243346 | n/a / better / better / better |
| 8.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-LloydMax | n/a | 0.000265424 [0.000209625, 0.000321223] | 0.258106 | 0.250654 | n/a / equiv / better / better |
| 8.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | n/a | 0.000530657 [0.000412106, 0.000649208] | 0.148075 | 0.12396 | n/a / equiv / better / better |
| 8.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | n/a | 0.000529241 [0.000410873, 0.000647609] | 0.148327 | 0.124189 | n/a / equiv / better / better |
| 8.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-LloydMax | n/a | 0.000259309 [0.000204701, 0.000313918] | 0.275693 | 0.268641 | n/a / equiv / better / better |
| 8.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | n/a | 0.000524542 [0.000407014, 0.000642071] | 0.158164 | 0.132855 | n/a / equiv / better / better |
| 8.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | n/a | 0.000523126 [0.000405787, 0.000640466] | 0.158433 | 0.1331 | n/a / equiv / better / better |
| 8.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-LloydMax | n/a | 0.000906818 [0.000692852, 0.00112078] | 0.307357 | 0.278218 | n/a / better / better / better |
| 8.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | n/a | 0.00147275 [0.0011978, 0.0017477] | 0.220736 | 0.121389 | n/a / better / better / better |
| 8.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | n/a | 0.00147087 [0.00119508, 0.00174666] | 0.220931 | 0.122095 | n/a / better / better / better |
| 8.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-LloydMax | n/a | 0.000841599 [0.000638843, 0.00104436] | 0.352631 | 0.321649 | n/a / equiv / better / better |
| 8.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | n/a | 0.00140753 [0.00114378, 0.00167127] | 0.253251 | 0.140338 | n/a / better / better / better |
| 8.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | n/a | 0.00140565 [0.00114106, 0.00167024] | 0.253474 | 0.141154 | n/a / better / better / better |
| 64.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-LloydMax | n/a | 0.000201504 [0.000196557, 0.000206451] | 0.141931 | 0.0575944 | n/a / equiv / better / better |
| 64.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | n/a | 0.000306357 [0.00029532, 0.000317395] | 0.128204 | 0.0246884 | n/a / equiv / better / better |
| 64.0 | 5 vs 4 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | n/a | 0.000305921 [0.00029435, 0.000317492] | 0.12825 | 0.0247837 | n/a / equiv / better / better |
| 64.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-LloydMax | n/a | 0.000200275 [0.000195177, 0.000205374] | 0.151353 | 0.0616973 | n/a / equiv / better / better |
| 64.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | n/a | 0.000305129 [0.000293883, 0.000316375] | 0.136715 | 0.0264472 | n/a / equiv / better / better |
| 64.0 | 5 vs 4 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | n/a | 0.000304692 [0.000292924, 0.000316461] | 0.136763 | 0.0265492 | n/a / equiv / better / better |
| 64.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-LloydMax | n/a | 0.00202671 [0.00198187, 0.00207156] | 0.160901 | 0.0378429 | n/a / better / better / better |
| 64.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-GaussianCDF | n/a | 0.00174579 [0.00168422, 0.00180737] | 0.200008 | 0.0427239 | n/a / better / better / better |
| 64.0 | 4 vs 3 bits | TQ-Uniform-Midrise | TQ-Sigmoid-Vasicek | n/a | 0.00174805 [0.00168612, 0.00180999] | 0.199892 | 0.0427313 | n/a / better / better / better |
| 64.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-LloydMax | n/a | 0.00200147 [0.00195785, 0.00204509] | 0.185217 | 0.047814 | n/a / better / better / better |
| 64.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-GaussianCDF | n/a | 0.00172055 [0.00165758, 0.00178353] | 0.230235 | 0.053981 | n/a / better / better / better |
| 64.0 | 4 vs 3 bits | TQ-Uniform-Midtread | TQ-Sigmoid-Vasicek | n/a | 0.00172281 [0.0016595, 0.00178612] | 0.230101 | 0.0539903 | n/a / better / better / better |

## Reproducibility and baseline audit

synthetic: 90 prior 3/4-bit rows compared; maximum non-timing metric delta 0.

niah: 90 prior 3/4-bit rows compared; maximum non-timing metric delta 0.
