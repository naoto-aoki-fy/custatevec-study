#include "device_assignment.hpp"

#include <cassert>
#include <map>
#include <string>

using namespace bench::device_assignment;

template<class F> void rejects(F f) {
    bool threw = false;
    try { f(); } catch (const std::runtime_error&) { threw = true; }
    assert(threw);
}

int main() {
    std::map<std::string, std::string> environment;
    const auto get = [&](const char* name) -> const char* {
        const auto found = environment.find(name);
        return found == environment.end() ? nullptr : found->second.c_str();
    };
    const auto isolated = discover(1, get);
    assert(isolated.device == 0 && isolated.local_rank == -1);
    // Distinct per-process GPU visibility permits any actual local rank.
    validate(isolated, 0);
    validate(isolated, 7);
    rejects([&] { discover(0, get); });
    rejects([&] { discover(4, get); });

    for (const char* source : {"OMPI_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID"}) {
        for (int rank = 0; rank < 4; ++rank) {
            environment = {{source, std::to_string(rank)}};
            const auto shared = discover(4, get);
            assert(shared.device == rank && shared.source == source);
            validate(shared, rank);
            rejects([&] { validate(shared, (rank + 1) % 4); });
            const auto per_process = discover(1, get);
            assert(per_process.device == 0);
            validate(per_process, rank);
            rejects([&] { validate(per_process, rank + 1); });
        }
        environment = {{source, "4"}};
        rejects([&] { discover(4, get); });
        for (const char* value : {"", "-1", "+1", " 1", "1 ", "1x", "2147483648",
                                  "999999999999999999999999999999999999"}) {
            environment = {{source, value}};
            rejects([&] { discover(4, get); });
            rejects([&] { discover(1, get); });
        }
    }
    assert(parse_rank("2147483647", "rank") == std::numeric_limits<int>::max());
    assert(parse_rank("0002", "rank") == 2);
    rejects([] { parse_rank(nullptr, "rank"); });

    // An enclosing Slurm allocation must not override Open MPI process ranks.
    environment = {{"OMPI_COMM_WORLD_LOCAL_RANK", "2"}, {"SLURM_LOCALID", "0"}};
    const auto nested = discover(4, get);
    assert(nested.device == 2 && nested.source == "OMPI_COMM_WORLD_LOCAL_RANK");
    validate(nested, 2);
    environment["SLURM_LOCALID"] = "not-a-process-rank";
    assert(discover(4, get).device == 2);
    environment["OMPI_COMM_WORLD_LOCAL_RANK"] = "bad";
    rejects([&] { discover(4, get); });

    auto inconsistent = nested;
    inconsistent.device = 0;
    rejects([&] { validate(inconsistent, 2); });
}
