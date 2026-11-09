#include "qcs.h"
#include <string.h>

int qcs_c_caller_checks(qcs_simulator* sim) {
    bit_t count = 0, bits[3] = {1, 1, 1};
    char text[4];
    if (!qcs_simulator_set_num_clbits(sim, 3) ||
        !qcs_simulator_get_num_clbits(sim, &count) || count != 3 ||
        !qcs_simulator_get_clbits(sim, bits) || bits[0] || bits[1] || bits[2] ||
        !qcs_simulator_get_clbits_string(sim, text) || strcmp(text, "000")) return 0;
    if (qcs_simulator_read(sim, 3, &count) != 0) return 0;
    if (qcs_simulator_gate_h(sim, NULL, 0, NULL, 0, NULL, 0) != 0) return 0;
    return qcs_simulator_reset_clbits(sim);
}
