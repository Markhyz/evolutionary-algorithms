#ifndef GUARD_H_EVO_ALG_MACROS
#define GUARD_H_EVO_ALG_MACROS

#include <cassert>
#include <memory>
#include <type_traits>

#define POINTER_ALIAS(type)                                                                                            \
    using unique_ptr = std::unique_ptr<type>;                                                                          \
    using shared_ptr = std::shared_ptr<type>;                                                                          \
    using const_unique_ptr = std::unique_ptr<type const>;                                                              \
    using const_shared_ptr = std::shared_ptr<type const>;

#define IS_DERIVED(derived, base) typename = std::enable_if_t<std::is_base_of_v<base, derived>>

#endif