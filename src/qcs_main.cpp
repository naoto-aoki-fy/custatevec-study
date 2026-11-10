// Adapted from rqs-svg 10a4da5b29a65f3eec0f556344ce06c3ff24072a.
// Changes: standard formatting, checked loading/API calls, validation and cleanup.
#include <cstdlib>
#include <cstdio>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <memory>

#include <limits>

#include <qcs.h>
#include "qcs_args.h"

static bool circuit_api_failed = false;

static qcs_initial_state parse_initial_state(const char *name)
{
    if (std::strcmp(name, "zero") == 0) return QCS_INITIAL_STATE_ZERO;
    if (std::strcmp(name, "flat") == 0) return QCS_INITIAL_STATE_FLAT;
    if (std::strcmp(name, "entangled") == 0) return QCS_INITIAL_STATE_ENTANGLED;
    if (std::strcmp(name, "sequential") == 0) return QCS_INITIAL_STATE_SEQUENTIAL;
    if (std::strcmp(name, "use_curand") == 0) return QCS_INITIAL_STATE_USE_CURAND;
    if (std::strcmp(name, "load_statevector") == 0) return QCS_INITIAL_STATE_LOAD_STATEVECTOR;
    throw std::invalid_argument(std::string("invalid --initial-state: ") + name +
        " (expected zero, flat, entangled, sequential, use_curand, load_statevector)");
}

static std::vector<bit_num_t> parse_mapping_csv(std::string const &mapping_text)
{
    if (mapping_text.empty() || mapping_text.back() == ',')
    {
        throw std::runtime_error("mapping string must not be empty or end with a comma");
    }

    std::vector<bit_num_t> mapping;
    std::stringstream ss(mapping_text);
    std::string token;
    while (std::getline(ss, token, ','))
    {
        if (token.empty())
        {
            throw std::runtime_error("mapping contains an empty entry");
        }
        size_t consumed = 0;
        int value = std::stoi(token, &consumed);
        if (consumed != token.size() || value < 0 || value > std::numeric_limits<bit_num_t>::max())
            throw std::runtime_error("invalid mapping entry: " + token);
        mapping.push_back(static_cast<bit_num_t>(value));
    }

    return mapping;
}

static std::vector<bit_num_t> invert_mapping(std::vector<bit_num_t> const &perm_p2l, bit_num_t const num_qubits)
{
    if ((int)perm_p2l.size() != num_qubits)
    {
        throw std::runtime_error("mapping size " + std::to_string(perm_p2l.size()) + " does not match num_qubits " + std::to_string(num_qubits));
    }

    std::vector<bit_num_t> perm_l2p(num_qubits, -1);
    for (int physical_qubit_num = 0; physical_qubit_num < num_qubits; physical_qubit_num++)
    {
        int const logical_qubit_num = perm_p2l[physical_qubit_num];
        if (logical_qubit_num < 0 || logical_qubit_num >= num_qubits)
        {
            throw std::runtime_error("mapping value " + std::to_string(logical_qubit_num) + " is out of range [0, " + std::to_string(num_qubits) + ")");
        }
        if (perm_l2p[logical_qubit_num] != -1)
        {
            throw std::runtime_error("mapping value " + std::to_string(logical_qubit_num) + " appears multiple times");
        }
        perm_l2p[logical_qubit_num] = physical_qubit_num;
    }

    return perm_l2p;
}

int main(int argc, char **argv)
try
{
    gengetopt_args_info parsed_options;
    if (cmdline_parser(argc, argv, &parsed_options) != 0)
    {
        throw std::runtime_error("failed to parse command-line options");
    }
    std::unique_ptr<gengetopt_args_info, void (*)(gengetopt_args_info *)> parsed_options_cleanup(&parsed_options, cmdline_parser_free);

    if (parsed_options.inputs_num != 1)
    {
        throw std::runtime_error("user circuit shared object path is required");
    }

    qcs_initial_state const initial_state = parse_initial_state(parsed_options.initial_state_arg);

    qcs_set_exception_callback([](char const *message, size_t length)
    {
        circuit_api_failed = true;
        fprintf(stderr, "[qcs pid=%ld] ", (long)getpid());
        fwrite(message, 1, length, stderr);
        fputc('\n', stderr);
        fflush(stderr);
    });

    std::string const usercircuit_so_path = parsed_options.inputs[0];
    int const num_samples = parsed_options.num_samples_arg;
    std::string const output_statevector_path = parsed_options.output_statevector_given > 0
                                                    ? parsed_options.output_statevector_arg
                                                    : "";
    if (num_samples <= 0)
    {
        throw std::runtime_error("num_samples must be greater than 0");
    }

    char const *const usercircuit_so_abspath = realpath(usercircuit_so_path.c_str(), NULL);
    if (usercircuit_so_abspath == NULL)
    {
        int const error = errno;
        if (error == ENOENT || error == ENOTDIR)
        {
            throw std::runtime_error("Circuit binary not found: '" + usercircuit_so_path + "'");
        }
        throw std::runtime_error("Failed to load circuit binary '" + usercircuit_so_path + "': " + std::strerror(error));
    }
    std::unique_ptr<char, void (*)(void *)> usercircuit_so_abspath_cleanup((char *)usercircuit_so_abspath, free);

    void *usercircuit_dl = dlopen(usercircuit_so_abspath, RTLD_NOW);
    if (usercircuit_dl == NULL)
    {
        char const *const error = dlerror();
        throw std::runtime_error("Failed to load circuit binary '" + usercircuit_so_path + "': " + error);
    }
    std::unique_ptr<void, int (*)(void *)> usercircuit_dl_cleanup(usercircuit_dl, dlclose);

    auto load_symbol = [&](const char *name) {
        dlerror();
        void *symbol = dlsym(usercircuit_dl, name);
        const char *error = dlerror();
        if (error || !symbol)
            throw std::runtime_error(std::string("Missing required circuit symbol '") + name + "': " + (error ? error : "null address"));
        return reinterpret_cast<void (*)(qcs_simulator *)>(symbol);
    };
    auto circuit_init = load_symbol("circuit_init");
    auto circuit_run = load_symbol("circuit_run");

    qcs_simulator *sim_ptr = nullptr;
    if (!qcs_simulator_create(&sim_ptr))
    {
        throw std::runtime_error("qcs_simulator_create failed");
    }
    std::unique_ptr<qcs_simulator, int (*)(qcs_simulator *)> sim(sim_ptr, qcs_simulator_destroy);

    circuit_init(sim.get());
    if (circuit_api_failed) throw std::runtime_error("circuit_init reported a QCS API failure");

    std::vector<bit_num_t> mapping;
    if (parsed_options.mapping_given > 0)
    {
        mapping = parse_mapping_csv(parsed_options.mapping_arg);
    }
    else
    {
        int const num_qubits = qcs_simulator_get_num_qubits_cxx(sim.get());
        mapping.resize(num_qubits);
        for (bit_num_t qubit_num = 0; qubit_num < num_qubits; qubit_num++)
        {
            mapping[qubit_num] = qubit_num;
        }
    }

    if (parsed_options.reversed_mapping_flag)
    {
        int const num_qubits = qcs_simulator_get_num_qubits_cxx(sim.get());
        mapping = invert_mapping(mapping, num_qubits);
    }
    int const num_qubits = qcs_simulator_get_num_qubits_cxx(sim.get());
    invert_mapping(mapping, num_qubits); // Validate before narrowing the count.
    qcs_simulator_set_mapping_cxx(sim.get(), mapping.data(), static_cast<bit_num_t>(mapping.size()));

    if (!qcs_simulator_allocate_memory_with_initial_state(sim.get(), initial_state))
    {
        throw std::runtime_error("qcs_simulator_allocate_memory failed");
    }
    if (!qcs_simulator_set_zero_state(sim.get()))
    {
        throw std::runtime_error("state initialization failed");
    }

    bit_t event_1 = 0;
    bit_t event_2 = 0;
    if (!qcs_simulator_event_create(sim.get(), &event_1) || !qcs_simulator_event_create(sim.get(), &event_2))
    {
        throw std::runtime_error("qcs_simulator_event_create failed");
    }

    for (int sample_num = 0; sample_num < num_samples; sample_num++)
    {

        qcs_simulator_event_record_cxx(sim.get(), event_1);

        circuit_run(sim.get());
        if (circuit_api_failed) throw std::runtime_error("circuit_run reported a QCS API failure");

        qcs_simulator_event_record_cxx(sim.get(), event_2);

        double elapsed_time = 0.0;
        if (!qcs_simulator_event_get_elapsed_time(sim.get(), event_1, event_2, &elapsed_time))
        {
            throw std::runtime_error("qcs_simulator_event_get_elapsed_time failed");
        }

        int const num_clbits = qcs_simulator_get_num_clbits_cxx(sim.get());
        std::vector<char> clbits_string(num_clbits + 1);
        qcs_simulator_get_clbits_string_cxx(sim.get(), clbits_string.data());
        int const proc_num = qcs_simulator_get_proc_num_cxx(sim.get());
        if (proc_num == 0)
        {
            fprintf(stdout, "{\"sample_num\": %d, \"clbits\": \"%s\", \"elapsed_time\": %.18g}\n", sample_num, clbits_string.data(), elapsed_time);
            fflush(stdout);
        }

        if (sample_num != num_samples - 1)
        {
            qcs_simulator_reinitialize_mapping_cxx(sim.get());
            if (!qcs_simulator_set_zero_state(sim.get()))
            {
                throw std::runtime_error("state reinitialization failed");
            }
            qcs_simulator_reset_clbits_cxx(sim.get());
        }
    }

    if (!output_statevector_path.empty())
    {
        qcs_simulator_save_statevector_cxx(sim.get(), output_statevector_path.c_str());
    }

    if (!qcs_simulator_destroy(sim.release()))
        throw std::runtime_error("qcs_simulator_destroy failed");
    return 0;
}
catch (const std::exception &error)
{
    std::fprintf(stderr, "qcs: %s\n", error.what());
    return EXIT_FAILURE;
}
