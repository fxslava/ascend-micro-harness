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

## Sigmoid-Vasicek parity assessment

Prespecified noninferiority margins: Recall@128=0, rho=0.001. A one-sided 95% lower CI must exceed minus the margin; zero-variance and single-seed outcomes cannot establish statistical noninferiority. No multiplicity correction is applied. A nonsignificant paired t test does not establish parity.

Retrieval parity is unassessed: run --source niah.

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

## Does the 5-bit gap disappear?

The following counts use separate context/kappa paired tests at 5 bits. Significant means two-sided p<0.05, without multiplicity adjustment. A nonsignificant difference is not evidence that the gap disappeared. Ratios are ranges of stratum means relative to same-bit Lloyd-Max, not pooled effect estimates.

| Source | Regime | NMSE(key) / Lloyd range | Significant NMSE(key) strata | Significant NMSE(QK) strata | Significant rho strata | Significant recall strata |
|---|---|---|---|---|---|---|
| synthetic | TQ-GaussianCDF | 1.487–3.741 | 3/3 | 3/3 | 3/3 | n/a |
| synthetic | TQ-Sigmoid-Vasicek | 1.475–3.742 | 3/3 | 3/3 | 3/3 | n/a |
| synthetic | TQ-Uniform-Midrise | 0.548–1.143 | 3/3 | 3/3 | 3/3 | n/a |
| synthetic | TQ-Uniform-Midtread | 0.585–1.221 | 3/3 | 3/3 | 3/3 | n/a |
