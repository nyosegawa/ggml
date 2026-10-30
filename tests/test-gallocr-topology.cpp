// Allocates two graphs one after the other with one ggml_gallocr on the CPU and each GPU: graphs with as many nodes
// and leafs, of the same sizes, wired differently. The first plan puts every node of the first graph in the memory of
// the input a, which no node reads after the first; the second graph reads a again at its end, and gives a wrong
// result if the first plan is used for it. The three operations differ, so that no backend fuses them into one kernel
// that reads every input before it writes.

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include <cstdio>
#include <vector>

static const int64_t n = 64;

// (a + b) c - c, or (a + b) c - a for the second graph, with a = 1, b = 2 and c = 4.
static float compute(ggml_backend_t backend, ggml_gallocr_t galloc, bool second) {
    ggml_init_params params = { ggml_tensor_overhead()*16 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(params);
    ggml_tensor * a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    ggml_tensor * c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
    for (ggml_tensor * t : {a, b, c}) {
        ggml_set_input(t);
    }
    ggml_tensor * out = ggml_sub(ctx, ggml_mul(ctx, ggml_add(ctx, a, b), c), second ? a : c);
    ggml_set_output(out);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);

    float result = -1.0f;
    if (ggml_gallocr_alloc_graph(galloc, gf)) {
        const float values[3] = {1.0f, 2.0f, 4.0f};
        ggml_tensor * inputs[3] = {a, b, c};
        for (int i = 0; i < 3; i++) {
            std::vector<float> data(n, values[i]);
            ggml_backend_tensor_set(inputs[i], data.data(), 0, ggml_nbytes(inputs[i]));
        }
        if (ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS) {
            std::vector<float> data(n);
            ggml_backend_tensor_get(out, data.data(), 0, ggml_nbytes(out));
            result = data[0];
            for (float v : data) {
                if (v != result) {
                    result = -1.0f;
                }
            }
        }
    }
    ggml_free(ctx);
    return result;
}

int main(void) {
    ggml_backend_load_all();
    int failed = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
            continue;
        }
        ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
        if (backend == nullptr) {
            continue;
        }
        ggml_gallocr_t galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        const float first = compute(backend, galloc, false);
        const float second = compute(backend, galloc, true);
        const bool ok = first == 8.0f && second == 11.0f;
        printf("%s: first graph %g (want 8), second graph %g (want 11): %s\n", ggml_backend_dev_name(dev), first, second, ok ? "OK" : "FAIL");
        failed += ok ? 0 : 1;
        ggml_gallocr_free(galloc);
        ggml_backend_free(backend);
    }
    return failed == 0 ? 0 : 1;
}
