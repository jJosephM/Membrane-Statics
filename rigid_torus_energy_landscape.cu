#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>

#define PI 3.141592653589793
#define N_THETA 721 //make it odd to use the simpson rule
#define REAL double
#define MAX_ITER 500000
#define TOL 1e-3f //Changed tol from 1e-4f for small patch
#define LEARNING_RATE 5e-6f //Changed lr from 5e-6f for small patch


__device__ REAL L(REAL R, REAL r, REAL a0, REAL a1, REAL theta, int flag) { // flags correspond to 0:sigma 1:alpha and 2:beta
    REAL s, c;
    sincos(theta, &s, &c);  // computes both for cuda transcendentals

    REAL delta = a0 * s + a1 * sin(2*theta) + 1;
    REAL deltap = a0 * c + 2*a1 * cos(2*theta);
    REAL g = pow(r*(R+r*s),2);
    
    REAL sigma = 1/delta + delta;
    REAL alpha = pow(PI*deltap,2)/(12*delta) * (pow(r,-2) + (7*pow(PI*delta*s,2))/(20*g));
    REAL beta = (delta*(pow(r,2)+pow(R+r*s,2)) + (pow(PI,2)*pow(delta,3)*(1+pow(s,2)))/12)/g;

    if (flag == 0) return sqrt(g)*sigma;
    if (flag == 1) return sqrt(g)*alpha; 
    if (flag == 2) return sqrt(g)*beta;
    return sqrt(g)*(sigma + alpha + beta);
}


// __device__ REAL energy(REAL R, REAL r, REAL a0) {
//     REAL E = 0.0;
//     for (int j = 0; j < N_THETA; ++j) {
//         REAL theta = 2 * PI * j / N_THETA;
//         E += L(R,r,a0,theta);
//     }
//     return E * (2 * PI / N_THETA);
// }


// Composite Simpson’s rule over [0, 2π] using N_THETA samples (including both endpoints).
// Requirement: (N_THETA - 1) must be even (i.e., N_THETA is odd). If not, we fall back to trapezoid.
__device__ inline REAL simpson_weight(int j, int N) {
    // Endpoints: 1; odd indices: 4; even interior: 2 
    if (j == 0 || j == N - 1) return 1.0;
    return (j & 1) ? 4.0 : 2.0;
}
__device__ REAL energy(REAL R, REAL r, REAL a0, REAL a1, int flag) {
    const REAL a = 0.0;
    const REAL b = 2.0 * PI;
    // Guard: if Simpson not applicable, use trapezoidal as a fallback.
    if (((N_THETA - 1) & 1) != 0) {
        // Trapezoid rule over N_THETA samples (endpoints included)
        const REAL h = (b - a) / (REAL)(N_THETA - 1);
        REAL E = 0.0;
        for (int j = 0; j < N_THETA; ++j) {
            REAL theta = a + h * (REAL)j;
            REAL w = (j == 0 || j == N_THETA - 1) ? 0.5 : 1.0;
            E += w * L(R, r, a0, a1, theta, flag);
        }
        return h * E;
    }
    // Composite Simpson’s rule
    const REAL h = (b - a) / (REAL)(N_THETA - 1);
    REAL E = 0.0;
    for (int j = 0; j < N_THETA; ++j) {
        REAL theta = a + h * (REAL)j;
        E += simpson_weight(j, N_THETA) * L(R, r, a0, a1, theta, flag);
    }
    return (h / 3.0) * E;
}


__global__ void minimize_coeffs(REAL* R_vals, REAL* r_vals, REAL* output,
                                int nR, int nr) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nR * nr;
    if (id >= total) return;

    int i_r = id % nr;
    int i_R  = id / nr;

    REAL R = R_vals[i_R];
    REAL r = r_vals[i_r];

    // Only compute for r in the correct region
    if (r > (R - 1.0) || r <= 1.0){
        // {printf("Didn't compute for R=%.2f r=%.2f\n", (REAL)R, (REAL)r);
        return;}

    REAL a0 = 0.03;
    REAL a1 = 0.03;

    REAL h = 1e-6;
    REAL grad[2] = {0.0};

    // Gradient descent loop
    bool clamped = false;
    for (int iter = 0; iter < MAX_ITER; ++iter) {
        REAL E0 = energy(R, r, a0, a1, -1);

        grad[0] = (energy(R, r, a0 + h, a1, -1) - E0) / h;
        grad[1] = (energy(R, r, a0 , a1 + h, -1) - E0) / h;

        REAL norm = sqrt(grad[0]*grad[0] + grad[1]*grad[1]);
        if (norm < TOL) {printf("Broke norm loop at R=%.2f r=%.2f and iter=%.2f \n", (REAL)R, (REAL)r, (REAL)iter);
            break;}

        a0 -= LEARNING_RATE * grad[0];
        a1 -= LEARNING_RATE * grad[1];

        // Clamp coefficients to prevent runaway growth and unphysical sols
        if (a0 < (-r) || a0 > r) { 
            clamped = true;
            printf("Clamped at R=%.2f r=%.2f a0=%.2f and iter=%.2f \n", (REAL)R, (REAL)r, (REAL)a0, (REAL)iter);
            a0 = 0.03;
            printf("Now we have a0=%.2f \n", (REAL)a0);
            break;}
        if (a1 < (-r) || a1 > r) { 
            clamped = true;
            printf("Clamped at R=%.2f r=%.2f a1=%.2f and iter=%.2f \n", (REAL)R, (REAL)r, (REAL)a1, (REAL)iter);
            a1 = 0.03;
            printf("Now we have a1=%.2f \n", (REAL)a1);
            break;}
            
        if (iter == MAX_ITER - 1) printf("Reached max iteration at R=%.2f r=%.2f\n", (REAL)R, (REAL)r);
    }
    
    REAL E_final = energy(R, r, a0, a1,-1);
    REAL sigma_E = energy(R, r, a0, a1, 0); // for plotting the partial energies
    REAL alpha_E = energy(R, r, a0, a1, 1);
    REAL beta_E = energy(R, r, a0, a1, 2);

    // if (clamped || !isfinite(E_final)) return;
  
    int offset = id * 8;
    output[offset + 0] = R;
    output[offset + 1] = r;
    output[offset + 2] = a0;
    output[offset + 3] = a1;
    output[offset + 4] = E_final;
    output[offset + 5] = sigma_E;
    output[offset + 6] = alpha_E;
    output[offset + 7] = beta_E;
    printf("Energy at R=%.2f and r=%.2f is %.2f with a0=%.2f\n", (REAL)R, (REAL)r, (REAL)E_final, (REAL)a0);
}


int main() {
    const int nR = 146, nr = 146;
    const int total = nR * nr;

    REAL* R_vals = new REAL[nR];
    REAL* r_vals = new REAL[nr];
    REAL* output = new REAL[total * 8];
    
    REAL Rmin = 1.0f, Rmax = 30.0f, rmin = 1.0f, rmax = 30.0f;

    // const int nR = 146, nr = 146;
    // const int total = nR * nr;

    // REAL* R_vals = new REAL[nR];
    // REAL* r_vals = new REAL[nr];
    // REAL* output = new REAL[total * 7];
    
    // REAL Rmin = 1.0f, Rmax = 30.0f, rmin = 1.0f, rmax = 30.0f;
    for (int i = 0; i < nR; ++i) R_vals[i] = Rmin + i * (Rmax - Rmin) / (nR - 1);
    for (int i = 0; i < nr; ++i) r_vals[i] = rmin + i * (rmax - rmin) / (nr - 1);

    REAL *d_R, *d_r;
    REAL *d_out;
    cudaMalloc(&d_R, nR * sizeof(REAL));
    cudaMalloc(&d_r, nr * sizeof(REAL));
    cudaMalloc(&d_out, total * 8 * sizeof(REAL));

    cudaMemcpy(d_R, R_vals, nR * sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_r, r_vals, nr * sizeof(REAL), cudaMemcpyHostToDevice);

    int threads = 512;
    int blocks = (total + threads - 1) / threads;
    minimize_coeffs<<<blocks, threads>>>(d_R, d_r, d_out, nR, nr);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }


    cudaMemcpy(output, d_out, total * 8 * sizeof(REAL), cudaMemcpyDeviceToHost);

    std::ofstream file("R_vs_r.csv");
    file << "R,r,a0,a1,E,sigma,alpha,beta\n";
    for (int i = 0; i < total; ++i) {
        for (int j = 0; j < 8; ++j) {
            file << output[i * 8 + j];
            if (j < 7) file << ",";
            else file << "\n";
        }
    }
    file.close();


    delete[] R_vals; delete[] r_vals; delete[] output;
    cudaFree(d_R); cudaFree(d_r); cudaFree(d_out);

    return 0;
}
