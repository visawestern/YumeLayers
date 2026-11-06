// YumeLayers — Metal GPU-движок для матричного умножения.
// Шейдер компилируется из исходника (без внешних .metal файлов).
import Foundation
import Metal

public final class MetalEngine {
    public static let shared = MetalEngine()

    public let device: MTLDevice
    public let queue: MTLCommandQueue
    public let deviceName: String

    static let shaderSource = """
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

    let matmulPS: MTLComputePipelineState
    let matmulTiledPS: MTLComputePipelineState
    let gradWAtomicPS: MTLComputePipelineState
    let gradWReducePS: MTLComputePipelineState
    let gradWPlainPS: MTLComputePipelineState
    let addPS: MTLComputePipelineState
    let addBiasPS: MTLComputePipelineState
    let mulPS: MTLComputePipelineState
    let subPS: MTLComputePipelineState
    let smulPS: MTLComputePipelineState
    let tanhPS: MTLComputePipelineState
    let tanhGradPS: MTLComputePipelineState
    let sumColsPS: MTLComputePipelineState
    let qsamplePS: MTLComputePipelineState
    let denoisePS: MTLComputePipelineState
    let adamPS: MTLComputePipelineState

    // Свёрточный путь
    let im2colPS: MTLComputePipelineState
    let im2col3dPS: MTLComputePipelineState
    let col2imPS: MTLComputePipelineState
    let col2im3dPS: MTLComputePipelineState
    let col2imS1PS: MTLComputePipelineState
    let col2im3dS1PS: MTLComputePipelineState
    let upsample2xPS: MTLComputePipelineState
    let upsample3dPS: MTLComputePipelineState
    let downsample2xPS: MTLComputePipelineState
    let downsample3dPS: MTLComputePipelineState
    let biasRowPS: MTLComputePipelineState
    let biasRow3dPS: MTLComputePipelineState
    let biasRowGradPS: MTLComputePipelineState
    let biasRowGradAtomicPS: MTLComputePipelineState
    let convBackInputS1PS: MTLComputePipelineState
    let reluPS: MTLComputePipelineState
    let reluGradPS: MTLComputePipelineState
    let sigmoidPS: MTLComputePipelineState
    let sigmoidGradPS: MTLComputePipelineState
    let addInPlacePS: MTLComputePipelineState
    let msePS: MTLComputePipelineState

    // Вариационный путь
    let fmaPS: MTLComputePipelineState
    let expPS: MTLComputePipelineState
    let addConstPS: MTLComputePipelineState
    let expHalfPS: MTLComputePipelineState
    let split2PS: MTLComputePipelineState
    let merge2PS: MTLComputePipelineState
    let klPS: MTLComputePipelineState
    let randnPS: MTLComputePipelineState
    let vaeGradPS: MTLComputePipelineState
    // 8-канальный путь: слой + композиция
    let layeredGradPS: MTLComputePipelineState

    private static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
        if let l = try? device.makeLibrary(source: shaderSource, options: nil) {
            return l
        }
        yumeLog("⚠️ runtime-компиляция Metal недоступна — пробую внешний xcrun metal")
        // fallback: внешний компилятор не зависит от MTLCompilerService
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("yume_shaders_\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let metalFile = dir.appendingPathComponent("shaders.metal")
        try Data(shaderSource.utf8).write(to: metalFile)
        let airFile = dir.appendingPathComponent("shaders.air")
        let libFile = dir.appendingPathComponent("shaders.metallib")
        try runExternal("/usr/bin/xcrun", ["-sdk", "macosx", "metal", "-c", metalFile.path, "-o", airFile.path])
        try runExternal("/usr/bin/xcrun", ["metallib", airFile.path, "-o", libFile.path])
        let lib = try device.makeLibrary(URL: libFile)
        yumeLog("✅ Шейдеры откомпилированы внешним xcrun metal")
        return lib
    }

    private static func runExternal(_ path: String, _ args: [String]) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: path)
        p.arguments = args
        p.standardOutput = FileHandle.nullDevice
        p.standardError = FileHandle.nullDevice
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            throw NSError(domain: "Yume", code: 20,
                          userInfo: [NSLocalizedDescriptionKey:
                            "внешний \(path) завершился с кодом \(p.terminationStatus)"])
        }
    }

    convenience init() {
        guard let dev = MTLCreateSystemDefaultDevice() else {
            fatalError("Metal недоступен на этой системе")
        }
        self.init(device: dev)
    }

    /// Движок на конкретном устройстве (например, на встроенной iGPU, чтобы
    /// переложить часть независимой работы со внешней видеокарты).
    init(device dev: MTLDevice) {
        self.device = dev
        self.deviceName = dev.name
        self.queue = dev.makeCommandQueue()!
        do {
            let lib = try Self.makeLibrary(device: dev)
            func ps(_ name: String) throws -> MTLComputePipelineState {
                guard let fn = lib.makeFunction(name: name) else {
                    fatalError("шейдер не найден: \(name)")
                }
                return try dev.makeComputePipelineState(function: fn)
            }
            self.matmulPS = try ps("matmul_kernel")
            self.matmulTiledPS = try ps("matmul_tiled_kernel")
            self.gradWAtomicPS = try ps("gradw_atomic_kernel")
            self.gradWReducePS = try ps("gradw_reduce_kernel")
            self.gradWPlainPS = try ps("gradw_plain_kernel")
            self.addPS = try ps("add_kernel")
            self.addBiasPS = try ps("addbias_kernel")
            self.mulPS = try ps("mul_kernel")
            self.subPS = try ps("sub_kernel")
            self.smulPS = try ps("smul_kernel")
            self.tanhPS = try ps("tanh_kernel")
            self.tanhGradPS = try ps("tanhgrad_kernel")
            self.sumColsPS = try ps("sumcols_kernel")
            self.qsamplePS = try ps("qsample_kernel")
            self.denoisePS = try ps("denoise_kernel")
            self.adamPS = try ps("adam_kernel")
            self.im2colPS = try ps("im2col_kernel")
            self.im2col3dPS = try ps("im2col3d_kernel")
            self.col2imPS = try ps("col2im_kernel")
            self.col2im3dPS = try ps("col2im3d_kernel")
            self.col2imS1PS = try ps("col2im_s1_kernel")
            self.col2im3dS1PS = try ps("col2im3d_s1_kernel")
            self.upsample2xPS = try ps("upsample2x_kernel")
            self.upsample3dPS = try ps("upsample3d_kernel")
            self.downsample2xPS = try ps("downsample2x_kernel")
            self.downsample3dPS = try ps("downsample3d_kernel")
            self.biasRowPS = try ps("biasrow_kernel")
            self.biasRow3dPS = try ps("biasrow3d_kernel")
            self.biasRowGradPS = try ps("biasrowgrad_kernel")
            self.biasRowGradAtomicPS = try ps("biasrowgrad_atomic_kernel")
            self.convBackInputS1PS = try ps("conv_back_input_s1_kernel")
            self.reluPS = try ps("relu_kernel")
            self.reluGradPS = try ps("relugrad_kernel")
            self.sigmoidPS = try ps("sigmoid_kernel")
            self.sigmoidGradPS = try ps("sigmoidgrad_kernel")
            self.addInPlacePS = try ps("addinplace_kernel")
            self.msePS = try ps("mse_reduce_kernel")
            self.fmaPS = try ps("fma_kernel")
            self.expPS = try ps("exp_kernel")
            self.addConstPS = try ps("addconst_kernel")
            self.expHalfPS = try ps("exphalf_kernel")
            self.split2PS = try ps("split2_kernel")
            self.merge2PS = try ps("merge2_kernel")
            self.klPS = try ps("kl_reduce_kernel")
            self.randnPS = try ps("randn_kernel")
            self.vaeGradPS = try ps("vae_grad_kernel")
            self.layeredGradPS = try ps("layered_grad_kernel")
        } catch {
            fatalError("ошибка компиляции Metal-шейдеров: \(error)")
        }
    }

    /// C = A (m×k) × B (k×n). Возвращает row-major [Float].
    func multiply(_ a: [Float], _ b: [Float], m: Int, k: Int, n: Int) -> [Float] {
        // Форма входов проверяется явно: `makeBuffer(bytes:length:)` копирует
        // ровно m·k·4 байт, и с коротким массивом это было чтение за границей.
        precondition(a.count == m * k, "multiply: a.count \(a.count) != m·k = \(m * k)")
        precondition(b.count == k * n, "multiply: b.count \(b.count) != k·n = \(k * n)")
        guard m > 0, k > 0, n > 0,
              let abuf = device.makeBuffer(bytes: a, length: m * k * 4, options: []),
              let bbuf = device.makeBuffer(bytes: b, length: k * n * 4, options: []),
              let cbuf = device.makeBuffer(length: m * n * 4, options: []) else {
            assertionFailure("multiply: не удалось выделить буферы — возвращаю нули")
            return [Float](repeating: 0, count: m * n)
        }
        guard let cmd = queue.makeCommandBuffer(),
              let enc = cmd.makeComputeCommandEncoder() else {
            return [Float](repeating: 0, count: m * n)
        }
        enc.setComputePipelineState(matmulPS)
        enc.setBuffer(abuf, offset: 0, index: 0)
        enc.setBuffer(bbuf, offset: 0, index: 1)
        enc.setBuffer(cbuf, offset: 0, index: 2)
        var mm = UInt32(m), kk = UInt32(k), nn = UInt32(n)
        enc.setBytes(&mm, length: 4, index: 3)
        enc.setBytes(&kk, length: 4, index: 4)
        enc.setBytes(&nn, length: 4, index: 5)
        var tA: UInt32 = 0, tB: UInt32 = 0
        enc.setBytes(&tA, length: 4, index: 6)
        enc.setBytes(&tB, length: 4, index: 7)
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        // ровно m×n нитей (современный API, избегает лишних потоков)
        let threads = MTLSize(width: m, height: n, depth: 1)
        enc.dispatchThreads(threads, threadsPerThreadgroup: tg)
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()
        let ptr = cbuf.contents().assumingMemoryBound(to: Float.self)
        var out = [Float](repeating: 0, count: m * n)
        for i in 0..<(m * n) { out[i] = ptr[i] }
        return out
    }
}
