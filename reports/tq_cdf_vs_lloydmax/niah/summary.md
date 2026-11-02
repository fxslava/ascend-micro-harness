# TurboQuant CDF vs. Lloyd-Max ablation

CPU synthetic evidence only; no CAModel or device latency claims.

128 coordinates, Pi = D H D, FP16 RMS scales; no FP8 intermediate. NIAH uses uniform 4:1 CSA reduction, 16 needles and 16 paired adversarial distractors (pre-compression cosine 0.70–0.82). Rank is 1-based against every compressed slot; ties prefer the smaller slot ID. FPR is the fraction of the 16 paired distractors in top-128.

Synthetic inputs are Gaussian with pre-rotation gains on channels 0,16,64,96. Synthetic retrieval fields are n/a. NIAH uses anisotropic unit vectors with gain 8. Spearman uses sampled backgrounds plus all 32 special slots. Each seed is one paired observation; needles are averaged within a seed. Contexts and kappa strata are never pooled for t tests.

GaussianCDF evaluates ndtr/ndtri. Sigmoid-Vasicek evaluates expit(1.5976*z + 0.070565*z^3) and logit plus three Newton iterations at every reconstructed coordinate. The requested label identifies this cubic logistic fit; no claim of historical attribution is made. Analytical paths do not use reconstruction lookup tables.

RoPE is a separate clean 64-channel half-mode probe, theta=160000. Phase error is the requested full-carrier acos angle, not an estimate of each pair's phase. Logit share follows DSV4's |rotary|/(|rotary|+|semantic|) convention. The counterfactual carrier is quantized; NIAH retrieval scores use content only.

stream_latency_s includes shared data generation/rotation/exact scores and the method's encode/pack/decode/score/rank/sample work. codec_us_per_slot isolates encode/pack/decode. Setup, warmup, final correlation/RoPE and diagnostic reductions are excluded. Method order is shuffled per chunk. Latency is host NumPy/SciPy wall time.

levels_used counts distinct decoded coordinates per vector (different group scales may produce more than 2^b values). code_levels_used in runs.csv counts distinct integer codes.

Seeds: 11,23,37,53,71; group size: per vector; chunk tokens: 8192; BLAS threads: 1.

## Mean metrics across seeds

| Source | Context / kappa | Bits | Regime | Recall@128 | Exact recall | Rank | Distractor FPR | Rho | Dilution | NMSE | Cosine | Levels |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| niah | 8192 | 3 | TQ-LloydMax | 0.55 | 0.5625 | 221.55 | 0.475 | 0.997895 | 0.993782 | 0.0312266 | 0.98434 | 7.99229 |
| niah | 8192 | 3 | TQ-GaussianCDF | 0.5375 | 0.5625 | 230.213 | 0.4625 | 0.99695 | 1.00686 | 0.0454373 | 0.977027 | 8 |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | 0.5375 | 0.5625 | 229.963 | 0.4625 | 0.99695 | 1.00935 | 0.045394 | 0.97704 | 8 |
| niah | 8192 | 4 | TQ-LloydMax | 0.5625 | 0.5625 | 223.675 | 0.475 | 0.999413 | 1.04575 | 0.00867781 | 0.995683 | 14.922 |
| niah | 8192 | 4 | TQ-GaussianCDF | 0.5625 | 0.5625 | 226.688 | 0.4625 | 0.998933 | 0.969845 | 0.0162001 | 0.991697 | 15.9944 |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | 0.5625 | 0.5625 | 226.6 | 0.4625 | 0.998933 | 0.971007 | 0.0161727 | 0.991711 | 15.9946 |
| niah | 32768 | 3 | TQ-LloydMax | 0.3625 | 0.4125 | 858.362 | 0.3 | 0.997927 | 0.926329 | 0.0312465 | 0.984337 | 7.99136 |
| niah | 32768 | 3 | TQ-GaussianCDF | 0.4 | 0.4125 | 890.288 | 0.2625 | 0.996992 | 0.98484 | 0.0454631 | 0.977014 | 7.99998 |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | 0.4 | 0.4125 | 890.087 | 0.2625 | 0.996993 | 0.984815 | 0.0454198 | 0.977027 | 7.99998 |
| niah | 32768 | 4 | TQ-LloydMax | 0.4 | 0.4125 | 847.663 | 0.275 | 0.999431 | 0.986544 | 0.0086919 | 0.995675 | 14.919 |
| niah | 32768 | 4 | TQ-GaussianCDF | 0.3875 | 0.4125 | 865.95 | 0.275 | 0.998952 | 0.994694 | 0.0162325 | 0.991679 | 15.9947 |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | 0.3875 | 0.4125 | 866.925 | 0.275 | 0.998953 | 0.994458 | 0.0162037 | 0.991694 | 15.9947 |
| niah | 131072 | 3 | TQ-LloydMax | 0.2 | 0.1875 | 3038.4 | 0.1 | 0.997942 | 0.98585 | 0.031222 | 0.984349 | 7.99203 |
| niah | 131072 | 3 | TQ-GaussianCDF | 0.175 | 0.1875 | 3103.03 | 0.075 | 0.997017 | 0.977823 | 0.0454036 | 0.977034 | 7.99998 |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | 0.175 | 0.1875 | 3102.15 | 0.075 | 0.997019 | 0.977935 | 0.0453603 | 0.977048 | 7.99998 |
| niah | 131072 | 4 | TQ-LloydMax | 0.1875 | 0.1875 | 2968.01 | 0.0875 | 0.999441 | 0.997847 | 0.00868175 | 0.995681 | 14.9173 |
| niah | 131072 | 4 | TQ-GaussianCDF | 0.2 | 0.1875 | 3031.11 | 0.075 | 0.998962 | 0.999415 | 0.0161931 | 0.991694 | 15.9951 |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | 0.2 | 0.1875 | 3031.36 | 0.075 | 0.998964 | 0.999406 | 0.0161648 | 0.991709 | 15.995 |

## RoPE and timing means

| Source | Context / kappa | Bits | Regime | RoPE share | Phase error deg | Stream s | us/slot | Codec us/slot | Effective bits/coord |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| niah | 8192 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.169132 | 82.5839 | 12.1282 | 3.125 |
| niah | 8192 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.172673 | 84.313 | 14.0072 | 3.125 |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.180082 | 87.9305 | 17.6311 | 3.125 |
| niah | 8192 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.168346 | 82.2004 | 11.9676 | 4.125 |
| niah | 8192 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.175609 | 85.7467 | 15.4519 | 4.125 |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.183019 | 89.3647 | 19.1381 | 4.125 |
| niah | 32768 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.671619 | 81.9848 | 10.5739 | 3.125 |
| niah | 32768 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.698936 | 85.3194 | 14.017 | 3.125 |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.738736 | 90.1777 | 18.8031 | 3.125 |
| niah | 32768 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.683061 | 83.3815 | 12.1097 | 4.125 |
| niah | 32768 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.712618 | 86.9895 | 15.6107 | 4.125 |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.753689 | 92.0031 | 20.6241 | 4.125 |
| niah | 131072 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 2.731 | 83.3435 | 10.7227 | 3.125 |
| niah | 131072 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 2.85375 | 87.0895 | 14.4815 | 3.125 |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 3.04866 | 93.0378 | 20.3485 | 3.125 |
| niah | 131072 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 2.82538 | 86.2236 | 13.5746 | 4.125 |
| niah | 131072 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 2.89029 | 88.2046 | 15.509 | 4.125 |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 3.08404 | 94.1175 | 21.4497 | 4.125 |

## Paired differences from Lloyd-Max

Two-sided paired t tests, unadjusted exploratory p-values; 95% paired CIs. Positive deltas mean the candidate metric is larger. Full statistics for every metric are in paired.csv. Undefined t tests are reported as n/a with their status.

| Source | Context / kappa | Bits | Regime | Metric | n | Delta | t | p | 95% CI | Status |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| niah | 8192 | 3 | TQ-GaussianCDF | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000944871 | -8.078 | 0.00127591 | [-0.00126963, -0.000620115] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0142107 | 95.425 | 7.23079e-08 | [0.0137973, 0.0146242] | ok |
| niah | 8192 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.87904 | 1.78223 | 0.149296 | [-1.04823, 4.80631] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000944908 | -8.10691 | 0.00125867 | [-0.00126852, -0.000621297] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0141674 | 94.9925 | 7.3633e-08 | [0.0137533, 0.0145815] | ok |
| niah | 8192 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 5.50295 | 9.50795 | 0.000683024 | [3.89602, 7.10988] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000479976 | -7.31284 | 0.00186002 | [-0.000662207, -0.000297745] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00752225 | 82.8143 | 1.27441e-07 | [0.00727005, 0.00777444] | ok |
| niah | 8192 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 3.48428 | 3.21476 | 0.0324415 | [0.475066, 6.49349] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0 | n/a | n/a | [0, 0] | zero_variance |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000479912 | -7.27233 | 0.00189934 | [-0.000663134, -0.00029669] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00749488 | 82.3866 | 1.30106e-07 | [0.0072423, 0.00774746] | ok |
| niah | 8192 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.17048 | 11.8888 | 0.000286672 | [5.49593, 8.84503] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | recall_at_128 | 5 | 0.0375 | 1.5 | 0.208 | [-0.0319111, 0.106911] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000934733 | -7.09809 | 0.00208064 | [-0.00130036, -0.000569109] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0142166 | 61.6215 | 4.15394e-07 | [0.0135761, 0.0148572] | ok |
| niah | 32768 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 3.44303 | 7.33325 | 0.00184059 | [2.13947, 4.7466] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0.0375 | 1.5 | 0.208 | [-0.0319111, 0.106911] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000933371 | -7.1361 | 0.00203933 | [-0.00129652, -0.000570224] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0141734 | 61.6784 | 4.13864e-07 | [0.0135353, 0.0148114] | ok |
| niah | 32768 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 8.22912 | 14.5464 | 0.000129887 | [6.65845, 9.7998] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000478985 | -6.74122 | 0.00252364 | [-0.00067626, -0.00028171] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00754063 | 53.417 | 7.35224e-07 | [0.00714869, 0.00793257] | ok |
| niah | 32768 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 3.50102 | 6.35225 | 0.00314696 | [1.97079, 5.03124] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.0125 | -1 | 0.373901 | [-0.0472056, 0.0222056] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000477611 | -6.7932 | 0.00245235 | [-0.000672816, -0.000282407] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.00751175 | 53.4137 | 7.35403e-07 | [0.00712129, 0.00790221] | ok |
| niah | 32768 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 8.51446 | 8.99671 | 0.000845019 | [5.88684, 11.1421] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | recall_at_128 | 5 | -0.025 | -1.63299 | 0.177808 | [-0.0675055, 0.0175055] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000924985 | -9.24998 | 0.000759425 | [-0.00120263, -0.000647345] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0141816 | 65.079 | 3.33968e-07 | [0.0135766, 0.0147866] | ok |
| niah | 131072 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 3.75885 | 9.05354 | 0.000824813 | [2.60613, 4.91158] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | -0.025 | -1.63299 | 0.177808 | [-0.0675055, 0.0175055] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000922796 | -9.23189 | 0.000765171 | [-0.00120032, -0.00064527] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0141383 | 65.1052 | 3.33431e-07 | [0.0135354, 0.0147412] | ok |
| niah | 131072 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 9.62579 | 25.8404 | 1.33239e-05 | [8.59154, 10.66] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.00047878 | -8.14881 | 0.00123419 | [-0.000641908, -0.000315651] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00751136 | 56.3625 | 5.93309e-07 | [0.00714134, 0.00788137] | ok |
| niah | 131072 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.93439 | 9.54748 | 0.000672177 | [1.37186, 2.49692] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | recall_at_128 | 5 | 0.0125 | 1 | 0.373901 | [-0.0222056, 0.0472056] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000477134 | -8.16477 | 0.00122503 | [-0.000639384, -0.000314884] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.007483 | 56.2532 | 5.97925e-07 | [0.00711367, 0.00785233] | ok |
| niah | 131072 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.87507 | 38.9357 | 2.59929e-06 | [7.31351, 8.43663] | ok |

## Sigmoid-Vasicek parity assessment

Prespecified noninferiority margins: Recall@128=0, rho=0.001. A one-sided 95% lower CI must exceed minus the margin; zero-variance and single-seed outcomes cannot establish statistical noninferiority. No multiplicity correction is applied. A nonsignificant paired t test does not establish parity.

Observed retrieval parity fails: at least one mean Recall@128 or rho loss exceeds its margin.
At least one stratum has a lower mean Recall@128 or rho than Lloyd-Max.
