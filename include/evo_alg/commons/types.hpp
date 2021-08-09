#ifndef GUARD_H_EVO_ALG_TYPES
#define GUARD_H_EVO_ALG_TYPES

#include <cstddef>
#include <tuple>

namespace evo_alg {
    namespace types {
        template <std::size_t N, typename... Ts>
        using NthType = typename std::tuple_element<N, std::tuple<Ts...>>::type;

        constexpr size_t invalid_index = (size_t) -1;
    }
}

#endif