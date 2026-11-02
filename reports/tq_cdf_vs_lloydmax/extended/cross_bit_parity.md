# Cross-bit parity: plain uniform versus Lloyd-Max / inverse CDF

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
