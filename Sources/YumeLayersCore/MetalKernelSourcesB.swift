// YumeLayers — MSL-ядра (часть B).
// Склеиваются в MetalEngine.
import Foundation

extension MetalKernels {
    static var part1: String {
    """
        uint y = rem / w;
        uint x = rem % w;
        uint w2 = w * 2;
        uint base = c * 4 * h * w + y * 2 * w2 + x * 2;
        Y[gid] = X[base] + X[base + 1] + X[base + w2] + X[base + w2 + 1];
    }

    // C = A + b по строкам (b длиной rows)
    kernel void biasrow_kernel(
        device const float* A [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] + b[gid / cols];
    }

    // ---------- Варианты без делений (3D-сетка) ----------

    kernel void col2im3d_kernel(
        device const float* C [[buffer(0)]],
        device float* G [[buffer(1)]],
        constant uint& ic [[buffer(2)]],
        constant uint& ih [[buffer(3)]],
        constant uint& iw [[buffer(4)]],
        constant uint& kh [[buffer(5)]],
        constant uint& kw [[buffer(6)]],
        constant uint& stride [[buffer(7)]],
        constant uint& pad [[buffer(8)]],
        constant uint& oh [[buffer(9)]],
        constant uint& ow [[buffer(10)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint x = gid.x, y = gid.y, c = gid.z;
        float acc = 0.0f;
        for (uint ky = 0; ky < kh; ++ky) {
            int sy = int(y) + int(pad) - int(ky);
            if (sy < 0 || sy % int(stride) != 0) continue;
            uint oy = uint(sy) / stride;
            if (oy >= oh) continue;
            for (uint kx = 0; kx < kw; ++kx) {
                int sx = int(x) + int(pad) - int(kx);
                if (sx < 0 || sx % int(stride) != 0) continue;
                uint ox = uint(sx) / stride;
                if (ox >= ow) continue;
                acc += C[(c * kh * kw + ky * kw + kx) * (oh * ow) + oy * ow + ox];
            }
        }
        G[(c * ih + y) * iw + x] = acc;
    }

    kernel void col2im3d_s1_kernel(
        device const float* C [[buffer(0)]],
        device float* G [[buffer(1)]],
        constant uint& ic [[buffer(2)]],
        constant uint& ih [[buffer(3)]],
        constant uint& iw [[buffer(4)]],
        constant uint& kh [[buffer(5)]],
        constant uint& kw [[buffer(6)]],
        constant uint& pad [[buffer(7)]],
        constant uint& oh [[buffer(8)]],
        constant uint& ow [[buffer(9)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint x = gid.x, y = gid.y, c = gid.z;
        float acc = 0.0f;
        for (uint ky = 0; ky < kh; ++ky) {
            int oy = int(y) + int(pad) - int(ky);
            if (oy < 0 || oy >= int(oh)) continue;
            for (uint kx = 0; kx < kw; ++kx) {
                int ox = int(x) + int(pad) - int(kx);
                if (ox < 0 || ox >= int(ow)) continue;
                acc += C[(c * kh * kw + ky * kw + kx) * (oh * ow) + uint(oy) * ow + uint(ox)];
            }
        }
        G[(c * ih + y) * iw + x] = acc;
    }

    kernel void upsample3d_kernel(
        device const float* X [[buffer(0)]],
        device float* Y [[buffer(1)]],
        constant uint& h [[buffer(2)]],
        constant uint& w [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint x = gid.x, y = gid.y, c = gid.z;   // координаты выхода
        uint w2 = w * 2;
        Y[(c * h * 2 + y) * w2 + x] = X[(c * h + y / 2) * w + x / 2];
    }

    kernel void downsample3d_kernel(
        device const float* X [[buffer(0)]],
        device float* Y [[buffer(1)]],
        constant uint& h [[buffer(2)]],
        constant uint& w [[buffer(3)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint x = gid.x, y = gid.y, c = gid.z;   // координаты выхода
        uint w2 = w * 2;
        uint base = (c * h * 2 + y * 2) * w2 + x * 2;
        Y[(c * h + y) * w + x] = X[base] + X[base + 1] + X[base + w2] + X[base + w2 + 1];
    }

    kernel void biasrow3d_kernel(
        device const float* A [[buffer(0)]],
        device const float* b [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint i = gid.y * cols + gid.x;
        C[i] = A[i] + b[gid.y];
    }

    // Прямой backprop входа свёртки: dx[c,y,x] = Σ_{oc,ky,kx} W[oc,c,ky,kx]·dOut[oc, y+pad-ky, x+pad-kx] (stride 1)
    // Заменяет материализацию dcols + col2im (шаг 1, без делений на stride).
    kernel void conv_back_input_s1_kernel(
        device const float* dOut [[buffer(0)]],
        device const float* W [[buffer(1)]],
        device float* dx [[buffer(2)]],
        constant uint& ic [[buffer(3)]],
        constant uint& oc [[buffer(4)]],
        constant uint& ih [[buffer(5)]],
        constant uint& iw [[buffer(6)]],
        constant uint& kh [[buffer(7)]],
        constant uint& kw [[buffer(8)]],
        constant uint& pad [[buffer(9)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint x = gid.x, y = gid.y, c = gid.z;
        // Размер ВЫХОДА свёртки, а не входа: раньше границы проверялись по
        // ih/iw, и при pad != (k-1)/2 читалось за пределами dOut (pad=0,
        // чтение до 2 лишних строк) либо терялись слагаемые градиента (pad=2).
        uint oh = ih + 2 * pad - kh + 1;
        uint ow = iw + 2 * pad - kw + 1;
        float acc = 0.0f;
        for (uint o = 0; o < oc; ++o) {
            uint woff = o * (ic * kh * kw) + c * (kh * kw);
            for (uint ky = 0; ky < kh; ++ky) {
                int oy = int(y) + int(pad) - int(ky);
                if (oy < 0 || oy >= int(oh)) continue;
                uint row = uint(oy) * ow;
                for (uint kx = 0; kx < kw; ++kx) {
                    int ox = int(x) + int(pad) - int(kx);
                    if (ox < 0 || ox >= int(ow)) continue;
                    acc += dOut[o * (oh * ow) + row + uint(ox)] * W[woff + ky * kw + kx];
                }
            }
        }
        dx[(c * ih + y) * iw + x] = acc;
    }

    // out[строка] = Σ по cols градиента
    kernel void biasrowgrad_kernel(
        device const float* G [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant uint& cols [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        float s = 0.0f;
        for (uint i = 0; i < cols; ++i) s += G[gid * cols + i];
        out[gid] = s;
    }

    // То же, но с разбиением cols на чанки и атомным накоплением — быстро при малом rows и огромном cols.
    kernel void biasrowgrad_atomic_kernel(
        device const float* G [[buffer(0)]],
        device atomic_float* out [[buffer(1)]],
        constant uint& cols [[buffer(2)]],
        constant uint& chunks [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint row = gid.y;
        uint ch = gid.x;
        uint per = (cols + chunks - 1) / chunks;
        uint start = ch * per;
        uint end = min(start + per, cols);
        float s = 0.0f;
        for (uint i = start; i < end; ++i) s += G[row * cols + i];
        atomic_fetch_add_explicit(&out[row], s, memory_order_relaxed);
    }

    kernel void relu_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = max(A[gid], 0.0f);
    }

    kernel void relugrad_kernel(
        device const float* Y [[buffer(0)]],
        device const float* G [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = Y[gid] > 0.0f ? G[gid] : 0.0f;
    }

    kernel void sigmoid_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = 1.0f / (1.0f + exp(-A[gid]));
    }

    kernel void sigmoidgrad_kernel(
        device const float* Z [[buffer(0)]],
        device const float* G [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        float s = 1.0f / (1.0f + exp(-Z[gid]));
        C[gid] = G[gid] * s * (1.0f - s);
    }

    kernel void addinplace_kernel(
        device float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        A[gid] += B[gid];
    }

    // Среднеквадратичная ошибка (MSE) одним проходом внутри одного threadgroup:
    // редукция суммы квадратов разностей, результат — один float в out[0].
    kernel void mse_reduce_kernel(
        device const float* recon [[buffer(0)]],
        device const float* target [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& n [[buffer(3)]],
        uint gid [[thread_position_in_grid]],
        uint lid [[thread_position_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]])
    {
        threadgroup float scratch[1024];
        float s = 0.0f;
        for (uint i = gid; i < n; i += tgSize) {
            float d = recon[i] - target[i];
            s += d * d;
        }
        scratch[lid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sz = tgSize / 2; sz > 0; sz >>= 1) {
            if (lid < sz) { scratch[lid] += scratch[lid + sz]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) { out[0] = scratch[0]; }
    }

    // ---------- Вариационный (VAE) путь ----------

    // C = A + B*C (сэмплирование: z = mu + sigma*eps)
    kernel void fma_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device const float* D [[buffer(2)]],
        device float* C [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] + B[gid] * D[gid];
    }

    kernel void exp_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = exp(A[gid]);
    }

    // C = A + s
    kernel void addconst_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        constant float& s [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] + s;
    }

    // logvar ограничен [-10, 10] во ВСЕХ операциях (экспонента и её градиент
    // должны совпадать, иначе численная проверка градиентов разойдётся).
    kernel void exphalf_kernel(
        device const float* LV [[buffer(0)]],
        device float* C [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = exp(0.5f * clamp(LV[gid], -10.0f, 10.0f));
    }

    // X (2C, HW) → A (C, HW), B (C, HW): первые C каналов — mu, вторые — logvar.
    kernel void split2_kernel(
        device const float* X [[buffer(0)]],
        device float* A [[buffer(1)]],
        device float* B [[buffer(2)]],
        constant uint& c [[buffer(3)]],
        constant uint& plane [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint x = gid.x;
        uint ch = gid.y;
        if (x >= plane) return;
        A[ch * plane + x] = X[ch * plane + x];
        B[ch * plane + x] = X[(ch + c) * plane + x];
    }

    // Обратное к split2: A, B (C, HW) → X (2C, HW).
    kernel void merge2_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* X [[buffer(2)]],
        constant uint& c [[buffer(3)]],
        constant uint& plane [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint x = gid.x;
        uint ch = gid.y;
        if (x >= plane) return;
        X[ch * plane + x] = A[ch * plane + x];
        X[(ch + c) * plane + x] = B[ch * plane + x];
    }

    // Сумма KL: 0.5 * Σ (mu² + exp(lv) - 1 - lv), один threadgroup.
    kernel void kl_reduce_kernel(
        device const float* MU [[buffer(0)]],
        device const float* LV [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant uint& n [[buffer(3)]],
        uint gid [[thread_position_in_grid]],
        uint lid [[thread_position_in_threadgroup]],
        uint tgSize [[threads_per_threadgroup]])
    {
        threadgroup float scratch[1024];
        float s = 0.0f;
        for (uint i = gid; i < n; i += tgSize) {
            float lv = clamp(LV[i], -10.0f, 10.0f);
            float mu = MU[i];
            s += 0.5f * (mu * mu + exp(lv) - 1.0f - lv);
        }
        scratch[lid] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint sz = tgSize / 2; sz > 0; sz >>= 1) {
            if (lid < sz) { scratch[lid] += scratch[lid + sz]; }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (lid == 0) { out[0] = scratch[0]; }
    }

    // Градиенты вариационного выхода.
    //   dmu = dz + βn·mu
    //   dlv = dz·ε·0.5·exp(0.5·lv) + βh·(exp(lv) - 1)
    // Первое слагаемое — перепараметризация (z = mu + σ·ε, σ = exp(0.5·lv)),
    // второе — KL-регуляризатор. exp(lv) и exp(0.5·lv) берутся от одного и
    // того же зажатого lv, иначе численная проверка градиентов разойдётся.
    kernel void vae_grad_kernel(
        device const float* DZ [[buffer(0)]],
        device const float* MU [[buffer(1)]],
        device const float* LV [[buffer(2)]],
        device const float* EPS [[buffer(3)]],
        device float* DMU [[buffer(4)]],
        device float* DLV [[buffer(5)]],
        constant float& betaN [[buffer(6)]],
        constant float& betaHalfN [[buffer(7)]],
        uint gid [[thread_position_in_grid]])
    {
        float lv = clamp(LV[gid], -10.0f, 10.0f);
        float dz = DZ[gid];
        DMU[gid] = dz + betaN * MU[gid];
        DLV[gid] = dz * EPS[gid] * 0.5f * exp(0.5f * lv) + betaHalfN * (exp(lv) - 1.0f);
    }

    // Детерминированный гауссов шум: одинаковый (seed, index) → одно и то же
    // значение. Нужен, чтобы обучение было воспроизводимо и чтобы численная
    // проверка градиентов использовала один и тот же eps для всех прогонов.
    static inline uint yume_hash(uint x) {
        x = ((x >> ((x >> 28u) + 4u)) ^ x) * 277803737u;
        return (x >> 22u) ^ x;
    }

    kernel void randn_kernel(
        device float* OUT [[buffer(0)]],
        constant uint& seed [[buffer(1)]],
        constant uint& n [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= n) return;
        uint h1 = yume_hash(gid * 747796405u + seed * 2891336453u + 1u);
        uint h2 = yume_hash(gid * 2654435761u + seed * 1013904223u + 0x9E3779B9u);
        float u1 = max(float(h1 >> 8) / 16777216.0f, 1e-7f);
        float u2 = float(h2 >> 8) / 16777216.0f;
        OUT[gid] = sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
    }

    // ---------- 8-канальный layered путь ----------
    //
    // RECON / TARGET / GRAD — (8, n) в CHW: канал c, пиксель i → c*n+i.
    // COMP — (3, n) в CHW: таргет собранного кадра.
    //
    // L = wl·MSE(8 каналов) + wc·MSE(clamp((bg·(1−m)+ob·m)·l), COMP)
    //
    // Считаем dL/dRECON сразу целиком за один проход на пиксель: без
    // промежуточных тензоров и без перекладки активаций на CPU. Это и есть
    // градиент, который уходит в сигмоиду выходного слоя d3.
    kernel void layered_grad_kernel(
        device const float* RECON [[buffer(0)]],
        device const float* TARGET [[buffer(1)]],
        device const float* COMP [[buffer(2)]],
        device float* GRAD [[buffer(3)]],
        constant uint& n [[buffer(4)]],
        constant float& wl [[buffer(5)]],
        constant float& wc [[buffer(6)]],
        uint gid [[thread_position_in_grid]])
    {
        if (gid >= n) return;
        uint i = gid;
        float m = RECON[6*n + i];
        float light = RECON[7*n + i];
        float im = 1.0f - m;
        float dl = 2.0f * wl / (8.0f * float(n));
        float dc = 2.0f * wc / (3.0f * float(n));
        float dM = 0.0f, dL = 0.0f;
        for (uint c = 0; c < 3; ++c) {
            float bg = RECON[c*n + i];
            float ob = RECON[(3+c)*n + i];
            float base = bg*im + ob*m;
            float v = base*light;
            float y = clamp(v, 0.0f, 1.0f);
            float e = y - COMP[c*n + i];
            // Производная clamp равна нулю вне (0,1); в рабочей области
            // значений сигмоиды она тождественно единица.
            float active = (v > 0.0f && v < 1.0f) ? 1.0f : 0.0f;
            float gc = dc * e * active;
            GRAD[c*n + i]     = dl*(bg - TARGET[c*n + i]) + gc*light*im;
            GRAD[(3+c)*n + i] = dl*(ob - TARGET[(3+c)*n + i]) + gc*light*m;
            dM += gc*light*(ob - bg);
            dL += gc*base;
        }
        GRAD[6*n + i] = dl*(m - TARGET[6*n + i]) + dM;
        GRAD[7*n + i] = dl*(light - TARGET[7*n + i]) + dL;
    }
    """
    }
}
