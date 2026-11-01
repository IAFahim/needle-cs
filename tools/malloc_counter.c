#define _GNU_SOURCE
#include <stddef.h>
#include <errno.h>

extern void *__libc_malloc(size_t size);
extern void *__libc_calloc(size_t count, size_t size);
extern void *__libc_realloc(void *block, size_t size);
extern void *__libc_memalign(size_t alignment, size_t size);

static volatile long long calls;

static void count_call(void)
{
    __atomic_fetch_add(&calls, 1, __ATOMIC_RELAXED);
}

long long needle_malloc_count(void)
{
    return __atomic_load_n(&calls, __ATOMIC_RELAXED);
}

void *malloc(size_t size)
{
    count_call();
    return __libc_malloc(size);
}

void *calloc(size_t count, size_t size)
{
    count_call();
    return __libc_calloc(count, size);
}

void *realloc(void *block, size_t size)
{
    count_call();
    return __libc_realloc(block, size);
}

void *memalign(size_t alignment, size_t size)
{
    count_call();
    return __libc_memalign(alignment, size);
}

void *aligned_alloc(size_t alignment, size_t size)
{
    count_call();
    return __libc_memalign(alignment, size);
}

int posix_memalign(void **result, size_t alignment, size_t size)
{
    count_call();
    void *block = __libc_memalign(alignment, size);
    if (block == NULL) return ENOMEM;
    *result = block;
    return 0;
}
