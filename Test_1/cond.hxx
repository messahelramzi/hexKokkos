#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosBlas1_nrm2.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_scal.hpp>  // for scale
#include <KokkosBlas2_gemv.hpp>   // dense matrix-vector multiply

using ExecDeviceSpace = Kokkos::DefaultExecutionSpace;
using ExecHostSpace = Kokkos::DefaultHostExecutionSpace;

// Read CSV file and return a tuple of (Kokkos::View A, Kokkos::View x,
// Kokkos::View y) Assumes A.csv is N x M matrix (rows x cols), x.csv is
// M-element vector Computes single-threaded y = A*x and returns it
template <typename Execspace, typename LayoutTag>
std::tuple<Kokkos::View<double**, LayoutTag, typename Execspace::memory_space>,
           Kokkos::View<double*, typename Execspace::memory_space>> 
           read_csv_files(const std::string& matrix_file, const std::string& vector_file) {
    using MemorySpace = typename Execspace::memory_space;
    Execspace execSpace{};

    // Read matrix from CSV
    std::vector<std::vector<double>> matrix_data;
    {
        std::ifstream file(matrix_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open matrix file: " + matrix_file);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::vector<double> row;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                row.push_back(std::stod(value));
            }
            matrix_data.push_back(row);
        }
    }

    // Read vector from CSV
    std::vector<double> vector_data;
    {
        std::ifstream file(vector_file);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open vector file: " + vector_file);
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            std::stringstream ss(line);
            std::string value;
            while (std::getline(ss, value, ',')) {
                vector_data.push_back(std::stod(value));
            }
        }
    }

    // Get dimensions
    const int N = matrix_data.size();                   // rows
    const int M = (N > 0) ? matrix_data[0].size() : 0;  // cols

    // Create Kokkos views
    auto A = Kokkos::View<double**, LayoutTag, MemorySpace>(
        Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "A"), N, M);
    auto b = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(execSpace, Kokkos::WithoutInitializing, "b"), M);

    // Copy data to host mirror, then to device
    auto A_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, A);
    auto b_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, b);

    Kokkos::parallel_for(
        "CopyMatrixToHost",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, N),
        [=](int i) {
            for (int j = 0; j < M; ++j) {
                A_host(i, j) = matrix_data[i][j];
                b_host(j) = vector_data[j];
            }
        });


    // Copy to device
    Kokkos::deep_copy(A, A_host);
    Kokkos::deep_copy(b, b_host);

    return std::make_tuple(A, b);
}

template <typename Execspace, typename AViewType>
Kokkos::View<double*, typename Execspace::memory_space> build_jacobi(
    const AViewType& A) {
    using MemorySpace = typename Execspace::memory_space;
    Kokkos::View<double*, MemorySpace> inv_diag("inv_diag", A.extent(0));

    Kokkos::parallel_for(
        "Jacobi", Kokkos::RangePolicy<Execspace>(0, A.extent(0)),
        KOKKOS_LAMBDA(const int i) {
            inv_diag(i) = double(1.0) / A(i, i);
        });

    return inv_diag;
}

template <typename AViewType, typename bViewType>
void apply_jacobi(
    const AViewType& inv_diag,
    const bViewType& r,
    bViewType& z) {
    using MemorySpace = typename AViewType::memory_space;
    using ExecSpace = typename AViewType::device_type::execution_space;
    Kokkos::parallel_for(
        "ApplyJacobi", Kokkos::RangePolicy<ExecSpace>(0, r.extent(0)),
        KOKKOS_LAMBDA(const int i) { z(i) = inv_diag(i) * r(i); });
}

template <typename AViewType, typename bViewType>
bViewType pcg_jacobi(const AViewType& A, const bViewType& b) {

    using MemorySpace = typename AViewType::memory_space;
    using Execspace = typename AViewType::device_type::execution_space;

    const auto N = b.extent(0);
    auto x = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(Execspace{}, Kokkos::WithoutInitializing, "x"), N);
    Kokkos::deep_copy(x, 0.0); // Initial guess x = 0

    Kokkos::View<double*, MemorySpace> r("r", N);
    Kokkos::View<double*, MemorySpace> z("z", N);
    Kokkos::View<double*, MemorySpace> p("p", N);
    Kokkos::View<double*, MemorySpace> Ap("Ap", N);

    // r = b - A x
    KokkosBlas::gemv("N", 1.0, A, x, 0.0, r);
    KokkosBlas::axpby(1.0, b, -1.0, r);

    auto inv_diag = build_jacobi<Execspace>(A);

    apply_jacobi(inv_diag, r, z);

    Kokkos::deep_copy(p, z);

    double rz_old = KokkosBlas::dot(r, z);
    double rnorm0 = KokkosBlas::nrm2(b);
    // double rnorm0 = KokkosBlas::nrm2(r);

    int iter = 0;

    const int max_iter = 1000;
    const double tol = 1e-8;

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A p
        KokkosBlas::gemv("N", 1.0, A, p, 0.0, Ap);

        double alpha = rz_old / KokkosBlas::dot(p, Ap);

        // x = x + alpha p
        KokkosBlas::axpy(alpha, p, x);

        // r = r - alpha Ap
        KokkosBlas::axpy(-alpha, Ap, r);

        double rnorm = KokkosBlas::nrm2(r);
        if (rnorm / rnorm0 < tol) {
            return x;
        }

        apply_jacobi(inv_diag, r, z);

        double rz_new = KokkosBlas::dot(r, z);
        double beta = rz_new / rz_old;

        // p = z + beta p
        KokkosBlas::axpby(1.0, z, beta, p);

        rz_old = rz_new;
    }

    return x;
}

template <typename AViewType, typename bViewType>
std::tuple<double, int> power_iteration(const AViewType& A, const bViewType& b) {

    using MemorySpace = typename AViewType::memory_space;
    using ExecSpace = typename AViewType::device_type::execution_space;
    
    const int max_iters = 1000;
        const double tol = 1e-6;            
    const int N = A.extent(0);
    const int M = A.extent(1);

    auto x = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing, "x"), N);
    auto Ax = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing, "Ax"), N);

    Kokkos::deep_copy(x, b); // Start with initial guess x = b

    double lambda_old = 0.0;
    double lambda_new = 0.0;

    int iter = 0;
    for (iter = 0; iter < max_iters; ++iter) {

        KokkosBlas::gemv("N", 1.0, A, x, 0.0, Ax);
        double norm_Ax = KokkosBlas::nrm2(Ax);
        KokkosBlas::scal(Ax, 1.0 / norm_Ax, Ax); // Normalize Ax to get the next x

        Kokkos::deep_copy(x, Ax); // Compute Ax for Rayleigh quotient

        KokkosBlas::gemv("N", 1.0, A, x, 0.0, Ax);
        lambda_new = KokkosBlas::dot(Ax, x);

        if (std::abs(lambda_new - lambda_old) < tol) {
            break; // Converged
        }
        lambda_old = lambda_new;
    }

    return std::make_tuple(lambda_new, std::min(iter, max_iters));
}

template <typename AViewType, typename bViewType>
std::tuple<double, int> inverse_power_iteration(const AViewType& A, const bViewType& b) {

    using MemorySpace = typename AViewType::memory_space;
    using ExecSpace = typename AViewType::device_type::execution_space;
    
    const int max_iters = 1000;
        const double tol = 1e-6;            
    const int N = A.extent(0);
    const int M = A.extent(1);

    auto x = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing, "x"), N);
    auto Ax = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing, "Ax"), N);
    auto y = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(ExecSpace{}, Kokkos::WithoutInitializing, "y"), N);

    Kokkos::deep_copy(x, b); // Start with initial guess x = b

    double lambda_old = 0.0;
    double lambda_new = 0.0;

    int iter = 0;
    for (iter = 0; iter < max_iters; ++iter) {

        y = pcg_jacobi(A, x); // Solve A y = x

        double norm_y = KokkosBlas::nrm2(y);
        KokkosBlas::scal(y, 1.0 / norm_y, y); // Normalize y to get the next y

        Kokkos::deep_copy(x, y); // Compute Ax for Rayleigh quotient
        KokkosBlas::gemv("N", 1.0, A, x, 0.0, Ax);

        lambda_new = KokkosBlas::dot(Ax, x);

        if (std::abs(lambda_new - lambda_old) < tol) {
            break; // Converged
        }
        lambda_old = lambda_new;
    }

    return std::make_tuple(lambda_new, std::min(iter, max_iters));
}
