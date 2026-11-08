// YumeLayers — MSL-ядра (часть A).
// Склеиваются в MetalEngine.
import Foundation

enum MetalKernels {
    static var part0: String {
    """
    #include <metal_stdlib>
    using namespace metal;

    kernel void matmul_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& M [[buffer(3)]],
        constant uint& K [[buffer(4)]],
        constant uint& N [[buffer(5)]],
        constant uint& transA [[buffer(6)]],
        constant uint& transB [[buffer(7)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint r = gid.x;
        uint c = gid.y;
        float acc = 0.0;
        for (uint k = 0; k < K; ++k) {
            float av = transA == 1 ? A[k * M + r] : A[r * K + k];
            float bv = transB == 1 ? B[c * K + k] : B[k * N + c];
            acc += av * bv;
        }
        C[r * N + c] = acc;
    }

    // Тайловый matmul 16×16 с общей памятью (в 10-20 раз быстрее наивного на больших матрицах)
    kernel void matmul_tiled_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& M [[buffer(3)]],
        constant uint& K [[buffer(4)]],
        constant uint& N [[buffer(5)]],
        constant uint& transA [[buffer(6)]],
        constant uint& transB [[buffer(7)]],
        uint2 tgid [[threadgroup_position_in_grid]],
        uint2 tid [[thread_position_in_threadgroup]])
    {
        threadgroup float As[16 * 16];
        threadgroup float Bs[16 * 16];

        uint tx = tid.x;
        uint ty = tid.y;
        uint rowC = tgid.x * 16 + ty;
        uint colC = tgid.y * 16 + tx;
        float acc = 0.0f;

        uint ktiles = (K + 15) / 16;
        for (uint kt = 0; kt < ktiles; ++kt) {
            uint k0 = kt * 16;
            uint ka = k0 + tx;
            float av = 0.0f;
            if (rowC < M && ka < K) {
                av = transA == 1 ? A[ka * M + rowC] : A[rowC * K + ka];
            }
            As[ty * 16 + tx] = av;

            // Для transB грузим тайл с транспонированием через общую память:
            // потоки идут вдоль K (коалесцированно), в shared кладём [c][k].
            if (transB == 1) {
                uint bcol = ty;
                uint bk = tx;
                uint bc = tgid.y * 16 + bcol;
                uint bkk = k0 + bk;
                float bv = 0.0f;
                if (bc < N && bkk < K) bv = B[bc * K + bkk];
                Bs[bcol * 16 + bk] = bv;
            } else {
                uint kb = k0 + ty;
                float bv = 0.0f;
                if (kb < K && colC < N) bv = B[kb * N + colC];
                Bs[ty * 16 + tx] = bv;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (transB == 1) {
                for (uint k = 0; k < 16; ++k) acc += As[ty * 16 + k] * Bs[tx * 16 + k];
            } else {
                for (uint k = 0; k < 16; ++k) acc += As[ty * 16 + k] * Bs[k * 16 + tx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (rowC < M && colC < N) {
            C[rowC * N + colC] = acc;
        }
    }

    // Градиент весов: C[r,c] = Σ_k A[r,k]·B[c,k] (по сути A·Bᵀ),
    // с разбиением K на чанки и атомным накоплением — быстро при малых M,N и огромном K.
    kernel void gradw_atomic_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device atomic_float* C [[buffer(2)]],
        constant uint& K [[buffer(3)]],
        constant uint& kk [[buffer(4)]],
        constant uint& chunks [[buffer(5)]],
        uint3 gid [[thread_position_in_grid]])
    {
        uint r = gid.x;
        uint c = gid.y;
        uint ch = gid.z;
        uint per = (K + chunks - 1) / chunks;
        uint start = ch * per;
        uint end = min(start + per, K);
        float acc = 0.0f;
        for (uint i = start; i < end; ++i) {
            acc += A[r * K + i] * B[c * K + i];
        }
        atomic_fetch_add_explicit(&C[r * kk + c], acc, memory_order_relaxed);
    }

    // Градиент весов: простой детерминированный вариант — один поток на элемент C.
    // Медленнее редукции, но без shared-памяти/барьеров (надёжно, без NaN).
    kernel void gradw_plain_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& K [[buffer(3)]],
        constant uint& kk [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint r = gid.x;
        uint c = gid.y;
        float acc = 0.0f;
        for (uint i = 0; i < K; ++i) {
            acc += A[r * K + i] * B[c * K + i];
        }
        C[r * kk + c] = acc;
    }

    // Градиент весов через групповую редукцию: одна группа (256 потоков) на элемент C,
    // без атомов — раз в 5-10 быстрее атомного варианта при большом K.
    kernel void gradw_reduce_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& K [[buffer(3)]],
        constant uint& kk [[buffer(4)]],
        uint2 gid [[thread_position_in_grid]],
        uint2 tid [[thread_position_in_threadgroup]])
    {
        uint r = gid.x;
        uint c = gid.y;
        const uint lanes = 256;
        threadgroup float sdata[256];
        uint per = (K + lanes - 1) / lanes;
        uint start = tid.x * per;
        uint end = min(start + per, K);
        float acc = 0.0f;
        for (uint i = start; i < end; ++i) {
            acc += A[r * K + i] * B[c * K + i];
        }
        sdata[tid.x] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // бинарная редукция в пределах группы
        for (uint stride = lanes / 2; stride > 0; stride >>= 1) {
            if (tid.x < stride) sdata[tid.x] += sdata[tid.x + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tid.x == 0) C[r * kk + c] = sdata[0];
    }

    kernel void add_kernel(        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] + B[gid];
    }

    kernel void addbias_kernel(
        device const float* A [[buffer(0)]],
        device const float* bias [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant uint& cols [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] + bias[gid % cols];
    }

    kernel void mul_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] * B[gid];
    }

    kernel void sub_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] - B[gid];
    }

    kernel void smul_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        constant float& s [[buffer(2)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = A[gid] * s;
    }

    kernel void qsample_kernel(
        device const float* A [[buffer(0)]],
        device const float* B [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant float& alpha [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = sqrt(alpha) * A[gid] + sqrt(1.0 - alpha) * B[gid];
    }

    kernel void denoise_kernel(
        device const float* Z [[buffer(0)]],
        device const float* pred [[buffer(1)]],
        device float* C [[buffer(2)]],
        constant float& alpha [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = (Z[gid] - (1.0 - alpha) * pred[gid]) / alpha;
    }

    kernel void tanh_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        C[gid] = tanh(A[gid]);
    }

    kernel void tanhgrad_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        uint gid [[thread_position_in_grid]])
    {
        float t = tanh(A[gid]);
        C[gid] = 1.0 - t * t;
    }

    kernel void sumcols_kernel(
        device const float* A [[buffer(0)]],
        device float* C [[buffer(1)]],
        device float* work [[buffer(2)]],
        constant uint& rows [[buffer(3)]],
        constant uint& cols [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        // gid: столбец (0..cols-1). Работаем по под-диапазонам строк.
        uint j = gid;
        uint chunk = max(1u, rows / 256u + 1u);
        float sum = 0.0;
        for (uint i = 0; i < rows; ++i) {
            sum += A[i * cols + j];
        }
        C[j] = sum;
    }

    kernel void adam_kernel(
        device float* param [[buffer(0)]],
        device const float* grad [[buffer(1)]],
        device float* m [[buffer(2)]],
        device float* v [[buffer(3)]],
        constant uint& t [[buffer(4)]],
        constant float& lr [[buffer(5)]],
        uint gid [[thread_position_in_grid]])
    {
        const float b1 = 0.9, b2 = 0.999, eps = 1e-8;
        float g = grad[gid];
        m[gid] = b1 * m[gid] + (1.0 - b1) * g;
        v[gid] = b2 * v[gid] + (1.0 - b2) * g * g;
        float mHat = m[gid] / (1.0 - pow(b1, float(t)));
        float vHat = v[gid] / (1.0 - pow(b2, float(t)));
        param[gid] -= lr * mHat / (sqrt(vHat) + eps);
    }

    // ---------- Свёрточный путь ----------

    // im2col: X (ic, ih, iw) → C ((ic*kh*kw), oh*ow)
    kernel void im2col_kernel(
        device const float* X [[buffer(0)]],
        device float* C [[buffer(1)]],
        constant uint& ic [[buffer(2)]],
        constant uint& ih [[buffer(3)]],
        constant uint& iw [[buffer(4)]],
        constant uint& kh [[buffer(5)]],
        constant uint& kw [[buffer(6)]],
        constant uint& stride [[buffer(7)]],
        constant uint& pad [[buffer(8)]],
        constant uint& oh [[buffer(9)]],
        constant uint& ow [[buffer(10)]],
        uint2 gid [[thread_position_in_grid]])
    {
        uint sp = gid.x;          // быстрая ось — пространство: запись подряд
        uint col = gid.y;
        uint karea = kh * kw;
        uint c = col / karea;
        uint r = col % karea;
        uint ky = r / kw;
        uint kx = r % kw;
        uint oy = sp / ow;
        uint ox = sp % ow;
        int iy = int(oy * stride + ky) - int(pad);
        int ix = int(ox * stride + kx) - int(pad);
        float v = 0.0f;
        if (iy >= 0 && iy < int(ih) && ix >= 0 && ix < int(iw)) {
            v = X[(c * ih + uint(iy)) * iw + uint(ix)];
        }
        C[col * (oh * ow) + sp] = v;
    }

    // im2col без целочисленных делений (3D-сетка: x=ox, y=oy, z=col)
    kernel void im2col3d_kernel(
        device const float* X [[buffer(0)]],
        device float* C [[buffer(1)]],
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
        uint ox = gid.x, oy = gid.y, col = gid.z;
        uint karea = kh * kw;
        uint c = col / karea;
        uint r = col % karea;
        uint ky = r / kw;
        uint kx = r % kw;
        int iy = int(oy * stride + ky) - int(pad);
        int ix = int(ox * stride + kx) - int(pad);
        float v = 0.0f;
        if (iy >= 0 && iy < int(ih) && ix >= 0 && ix < int(iw)) {
            v = X[(c * ih + uint(iy)) * iw + uint(ix)];
        }
        C[col * (oh * ow) + oy * ow + ox] = v;
    }

    // col2im: градиент ((ic*kh*kw), oh*ow) → градиент (ic, ih, iw)
    kernel void col2im_kernel(
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
        uint gid [[thread_position_in_grid]])
    {
        uint plane = ih * iw;
        uint c = gid / plane;
        uint rem = gid % plane;
        uint y = rem / iw;
        uint x = rem % iw;
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
        G[gid] = acc;
    }

    // col2im для stride=1 (без дорогого деления по модулю)
    kernel void col2im_s1_kernel(
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
        uint gid [[thread_position_in_grid]])
    {
        uint plane = ih * iw;
        uint c = gid / plane;
        uint rem = gid % plane;
        uint y = rem / iw;
        uint x = rem % iw;
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
        G[gid] = acc;
    }

    // Ближайший апсемплинг ×2: (c, h, w) → (c, 2h, 2w)
    kernel void upsample2x_kernel(
        device const float* X [[buffer(0)]],
        device float* Y [[buffer(1)]],
        constant uint& h [[buffer(2)]],
        constant uint& w [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        uint plane = h * w;
        uint c = gid / (4 * plane);
        uint rem = gid % (4 * plane);
        uint y = rem / (2 * w);
        uint x = rem % (2 * w);
        Y[gid] = X[(c * h + y / 2) * w + x / 2];
    }

    // Сумма 2×2 (обратный к upsample2x): (c, 2h, 2w) → (c, h, w)
    kernel void downsample2x_kernel(
        device const float* X [[buffer(0)]],
        device float* Y [[buffer(1)]],
        constant uint& h [[buffer(2)]],
        constant uint& w [[buffer(3)]],
        uint gid [[thread_position_in_grid]])
    {
        uint plane = h * w;
        uint c = gid / plane;
        uint rem = gid % plane;
    """
    }
}
