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
| synthetic | 1 | 3 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.981396 | n/a | 0.0339598 | 0.983102 | 7.99595 |
| synthetic | 1 | 3 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.970015 | n/a | 0.0566955 | 0.972695 | 8 |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.970035 | n/a | 0.0566267 | 0.972716 | 8 |
| synthetic | 1 | 4 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.994883 | n/a | 0.00931563 | 0.995413 | 15.2697 |
| synthetic | 1 | 4 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.988043 | n/a | 0.0226153 | 0.989163 | 15.9964 |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.988064 | n/a | 0.0225751 | 0.989183 | 15.9965 |
| synthetic | 8 | 3 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.998669 | n/a | 0.0308678 | 0.984484 | 7.99048 |
| synthetic | 8 | 3 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.998103 | n/a | 0.0429809 | 0.977801 | 7.99995 |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.998105 | n/a | 0.042943 | 0.977814 | 7.99995 |
| synthetic | 8 | 4 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.999632 | n/a | 0.00859433 | 0.995712 | 14.8396 |
| synthetic | 8 | 4 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.999367 | n/a | 0.0149806 | 0.99209 | 15.994 |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.999368 | n/a | 0.0149551 | 0.992104 | 15.9939 |
| synthetic | 64 | 3 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.997868 | n/a | 0.0318173 | 0.985272 | 7.34001 |
| synthetic | 64 | 3 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.998149 | n/a | 0.0255961 | 0.988072 | 7.81335 |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.998147 | n/a | 0.0256109 | 0.98807 | 7.81328 |
| synthetic | 64 | 4 | TQ-LloydMax | n/a | n/a | n/a | n/a | 0.999779 | n/a | 0.00831476 | 0.995964 | 12.361 |
| synthetic | 64 | 4 | TQ-GaussianCDF | n/a | n/a | n/a | n/a | 0.999674 | n/a | 0.00920502 | 0.995993 | 14.7498 |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | n/a | n/a | n/a | n/a | 0.999674 | n/a | 0.00920174 | 0.995992 | 14.7494 |

## RoPE and timing means

| Source | Context / kappa | Bits | Regime | RoPE share | Phase error deg | Stream s | us/slot | Codec us/slot | Effective bits/coord |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| synthetic | 1 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.251855 | 30.744 | 11.4056 | 3.125 |
| synthetic | 1 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.271154 | 33.0998 | 13.7977 | 3.125 |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.324782 | 39.6462 | 20.265 | 3.125 |
| synthetic | 1 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.261367 | 31.9052 | 12.4898 | 4.125 |
| synthetic | 1 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.272066 | 33.2112 | 13.89 | 4.125 |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.323053 | 39.4352 | 20.0711 | 4.125 |
| synthetic | 8 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.254178 | 31.0276 | 11.1698 | 3.125 |
| synthetic | 8 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.281398 | 34.3504 | 14.5637 | 3.125 |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.323178 | 39.4505 | 19.5254 | 3.125 |
| synthetic | 8 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.273435 | 33.3783 | 13.5561 | 4.125 |
| synthetic | 8 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.295518 | 36.074 | 16.2132 | 4.125 |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.338555 | 41.3275 | 21.4402 | 4.125 |
| synthetic | 64 | 3 | TQ-LloydMax | 0.436142 | 10.287 | 0.246033 | 30.0333 | 9.7363 | 3.125 |
| synthetic | 64 | 3 | TQ-GaussianCDF | 0.436142 | 12.9877 | 0.283839 | 34.6484 | 14.3495 | 3.125 |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | 0.436142 | 12.9821 | 0.33475 | 40.863 | 20.3702 | 3.125 |
| synthetic | 64 | 4 | TQ-LloydMax | 0.436142 | 5.3627 | 0.261152 | 31.8789 | 11.5703 | 4.125 |
| synthetic | 64 | 4 | TQ-GaussianCDF | 0.436142 | 7.88981 | 0.291833 | 35.6241 | 15.2646 | 4.125 |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | 0.436142 | 7.88161 | 0.331695 | 40.4902 | 20.0566 | 4.125 |

## Paired differences from Lloyd-Max

Two-sided paired t tests, unadjusted exploratory p-values; 95% paired CIs. Positive deltas mean the candidate metric is larger. Full statistics for every metric are in paired.csv. Undefined t tests are reported as n/a with their status.

| Source | Context / kappa | Bits | Regime | Metric | n | Delta | t | p | 95% CI | Status |
|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| synthetic | 1 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.0113809 | -70.6237 | 2.40862e-07 | [-0.0118284, -0.0109335] | ok |
| synthetic | 1 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0227357 | 349.601 | 4.0164e-10 | [0.0225551, 0.0229162] | ok |
| synthetic | 1 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.39211 | 3.72047 | 0.0204661 | [0.606971, 4.17726] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.011361 | -72.1974 | 2.20552e-07 | [-0.0117979, -0.0109241] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0226668 | 349.92 | 4.00178e-10 | [0.022487, 0.0228467] | ok |
| synthetic | 1 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 8.85935 | 8.86714 | 0.000893435 | [6.08534, 11.6334] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.00684004 | -137.543 | 1.6759e-08 | [-0.00697812, -0.00670197] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | nmse | 5 | 0.0132997 | 328.664 | 5.14179e-10 | [0.0131874, 0.0134121] | ok |
| synthetic | 1 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 1.4002 | 4.54115 | 0.0104879 | [0.544122, 2.25627] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.00681911 | -139.869 | 1.56718e-08 | [-0.00695447, -0.00668375] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0132595 | 330.987 | 4.99895e-10 | [0.0131483, 0.0133707] | ok |
| synthetic | 1 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.58133 | 6.19427 | 0.00345352 | [4.18317, 10.9795] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | spearman_rho | 5 | -0.000565927 | -23.2718 | 2.02071e-05 | [-0.000633446, -0.000498409] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | nmse | 5 | 0.0121131 | 335.368 | 4.74283e-10 | [0.0120128, 0.0122134] | ok |
| synthetic | 8 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 3.39394 | 3.75181 | 0.0199158 | [0.882328, 5.90555] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000564054 | -22.878 | 2.16257e-05 | [-0.000632507, -0.000495601] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0120752 | 335.292 | 4.74713e-10 | [0.0119752, 0.0121752] | ok |
| synthetic | 8 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 8.35561 | 5.09086 | 0.00702722 | [3.79864, 12.9126] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000265233 | -11.5652 | 0.000319295 | [-0.000328907, -0.000201559] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | nmse | 5 | 0.00638626 | 276.331 | 1.02895e-09 | [0.00632209, 0.00645042] | ok |
| synthetic | 8 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 2.6571 | 2.24224 | 0.0884024 | [-0.633049, 5.94725] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000263817 | -11.5386 | 0.000322179 | [-0.000327297, -0.000200337] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.0063608 | 273.82 | 1.06721e-09 | [0.00629631, 0.0064253] | ok |
| synthetic | 8 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 7.88408 | 15.1211 | 0.000111497 | [6.43645, 9.33171] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | spearman_rho | 5 | 0.00028092 | 7.40087 | 0.00177798 | [0.000175532, 0.000386307] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | nmse | 5 | -0.0062212 | -64.9141 | 3.37372e-07 | [-0.00648729, -0.00595511] | ok |
| synthetic | 64 | 3 | TQ-GaussianCDF | codec_us_per_slot | 5 | 4.61317 | 4.74031 | 0.00903515 | [1.91119, 7.31515] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | 0.000278663 | 7.32039 | 0.0018528 | [0.000172973, 0.000384353] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | nmse | 5 | -0.00620636 | -64.6182 | 3.4359e-07 | [-0.00647303, -0.00593969] | ok |
| synthetic | 64 | 3 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 10.6339 | 9.80376 | 0.000606795 | [7.62235, 13.6454] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | spearman_rho | 5 | -0.000104854 | -38.9773 | 2.58821e-06 | [-0.000112322, -9.73846e-05] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | nmse | 5 | 0.000890259 | 96.738 | 6.84627e-08 | [0.000864708, 0.00091581] | ok |
| synthetic | 64 | 4 | TQ-GaussianCDF | codec_us_per_slot | 5 | 3.69429 | 11.3381 | 0.00034498 | [2.78964, 4.59894] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | spearman_rho | 5 | -0.000104417 | -35.8184 | 3.6264e-06 | [-0.000112511, -9.63231e-05] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | nmse | 5 | 0.000886979 | 94.5321 | 7.50775e-08 | [0.000860928, 0.000913029] | ok |
| synthetic | 64 | 4 | TQ-Sigmoid-Vasicek | codec_us_per_slot | 5 | 8.48624 | 8.95755 | 0.000859299 | [5.85588, 11.1166] | ok |

## Sigmoid-Vasicek parity assessment

Prespecified noninferiority margins: Recall@128=0, rho=0.001. A one-sided 95% lower CI must exceed minus the margin; zero-variance and single-seed outcomes cannot establish statistical noninferiority. No multiplicity correction is applied. A nonsignificant paired t test does not establish parity.

Retrieval parity is unassessed: run --source niah.
