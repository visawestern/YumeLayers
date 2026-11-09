// YumeLayers — MSL-ядра (часть C): групповая нормализация.
// Склеиваются в MetalEngine.
import Foundation

extension MetalKernels {
    // Групповая нормализация (Этап 1). Один поток на группу каналов:
    // считает среднее/дисперсию по группе (группа × пространство) и
    // нормирует к N(0,1). Вход — CHW-тензор: rows = каналы, cols = плоскость.
    static let groupNormSource = """
    kernel void groupnorm_kernel(
        device const float* in [[buffer(0)]],
        device float* out [[buffer(1)]],
        constant int& rows [[buffer(2)]],
        constant int& cols [[buffer(3)]],
        constant int& groups [[buffer(4)]],
        uint gid [[thread_position_in_grid]]
    ) {
        int C = rows;
        int P = cols;
        if (C <= 0 || P <= 0 || groups <= 0) return;
        int gSize = C / groups;
        if (gSize <= 0) return;
        int g = (int)gid;
        if (g >= groups) return;
        int c0 = g * gSize;
        int n = gSize * P;
        float sum = 0.0f, sumSq = 0.0f;
        for (int c = c0; c < c0 + gSize; c++) {
            int base = c * P;
            for (int p = 0; p < P; p++) {
                float v = in[base + p];
                sum += v; sumSq += v * v;
            }
        }
        float mean = sum / (float)n;
        float var_ = (sumSq / (float)n) - mean * mean;
        if (var_ < 1e-5f) var_ = 1e-5f;
        float invStd = 1.0f / sqrt(var_);
        for (int c = c0; c < c0 + gSize; c++) {
            int base = c * P;
            for (int p = 0; p < P; p++) {
                int idx = base + p;
                out[idx] = (in[idx] - mean) * invStd;
            }
        }
    }

    // Градиент групповой нормализации: dx_i = (1/σ)·(dy_i − mean(dy) −
    // ((x_i − μ)/σ²)·mean(dy·(x − μ))). Один поток на группу; статистики
    // переиспользуются для всех элементов группы.
    kernel void groupnorm_grad_kernel(
        device const float* in [[buffer(0)]],
        device const float* dy [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant int& rows [[buffer(3)]],
        constant int& cols [[buffer(4)]],
        constant int& groups [[buffer(5)]],
        uint gid [[thread_position_in_grid]]
    ) {
        int C = rows;
        int P = cols;
        if (C <= 0 || P <= 0 || groups <= 0) return;
        int gSize = C / groups;
        if (gSize <= 0) return;
        int g = (int)gid;
        if (g >= groups) return;
        int c0 = g * gSize;
        int n = gSize * P;
        float sum = 0.0f, sumSq = 0.0f;
        for (int c = c0; c < c0 + gSize; c++) {
            int base = c * P;
            for (int p = 0; p < P; p++) {
                float v = in[base + p];
                sum += v; sumSq += v * v;
            }
        }
        float mean = sum / (float)n;
        float var_ = (sumSq / (float)n) - mean * mean;
        if (var_ < 1e-5f) var_ = 1e-5f;
        float invStd = 1.0f / sqrt(var_);
        float sumDy = 0.0f, sumDyXm = 0.0f;
        for (int c = c0; c < c0 + gSize; c++) {
            int base = c * P;
            for (int p = 0; p < P; p++) {
                int idx = base + p;
                float xm = in[idx] - mean;
                sumDy += dy[idx];
                sumDyXm += dy[idx] * xm;
            }
        }
        float meanDy = sumDy / (float)n;
        float meanDyXm = sumDyXm / (float)n;
        for (int c = c0; c < c0 + gSize; c++) {
            int base = c * P;
            for (int p = 0; p < P; p++) {
                int idx = base + p;
                float xm = in[idx] - mean;
                out[idx] = invStd * (dy[idx] - meanDy - xm * meanDyXm / var_);
            }
        }
    }
    // Диффузия: прямое зашумление латента x_t = √ᾱ·x0 + √(1−ᾱ)·ε.
    kernel void diffuse_kernel(
        device const float* x0 [[buffer(0)]],
        device const float* noise [[buffer(1)]],
        device float* out [[buffer(2)]],
        constant float& sqrtAlphaBar [[buffer(3)]],
        constant float& sqrtOneMinus [[buffer(4)]],
        uint gid [[thread_position_in_grid]])
    {
        out[gid] = sqrtAlphaBar * x0[gid] + sqrtOneMinus * noise[gid];
    }
    """
}
