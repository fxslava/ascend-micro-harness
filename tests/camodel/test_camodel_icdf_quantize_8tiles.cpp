// V3 runner: same checks as test_camodel_icdf_quantize.cpp, with the tile
// count raised past the depth-2 ring (ICDF_QUANT_TILES=8 comes from CMake),
// so every MTE2/MTE3 reuse back edge is contended at least once.
#include "test_camodel_icdf_quantize.cpp"
