#pragma once

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_dot.hpp>
#include <KokkosBlas1_axpby.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosSparse_CrsMatrix.hpp>

using ExecDeviceSpace = Kokkos::DefaultExecutionSpace;
using ExecHostSpace = Kokkos::DefaultHostExecutionSpace;

// Read CSV file and return a tuple of (Kokkos::View rowptr, Kokkos::View col,
// Kokkos::View val) of matrix A (csr format) and (Kokkos::View) of vector.
template <typename Execspace>
KokkosSparse::CrsMatrix<double, int, Execspace, void, int> read_csv_files() {
    using MemorySpace = typename Execspace::memory_space;
    using CSRMatrixType =
        KokkosSparse::CrsMatrix<double, int, Execspace, void, int>;

    Execspace execspace{};

    // Read rowptr from CSV
    std::vector<int> rowptr_data;
    {
        std::ifstream file("./rowptr.csv");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open rowptr file: ./rowptr.csv");
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            rowptr_data.push_back(std::stod(line));
        }
    }

    // Read col from CSV
    std::vector<int> col_data;
    {
        std::ifstream file("./col.csv");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open col file: ./col.csv");
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            col_data.push_back(std::stod(line));
        }
    }

    // Read val from CSV
    std::vector<double> val_data;
    {
        std::ifstream file("./val.csv");
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open val file: ./val.csv");
        }
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            val_data.push_back(std::stod(line));
        }
    }

    // Create Kokkos views
    auto rowptr = Kokkos::View<int*, MemorySpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "rowptr"),
        rowptr_data.size());
    auto col = Kokkos::View<int*, MemorySpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "col"),
        col_data.size());
    auto val = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(execspace, Kokkos::WithoutInitializing, "val"),
        val_data.size());

    // Copy data to host mirror, then to device
    auto rowptr_host =
        Kokkos::create_mirror_view(Kokkos::WithoutInitializing,
                                   Kokkos::DefaultHostExecutionSpace{}, rowptr);
    auto col_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, col);
    auto val_host = Kokkos::create_mirror_view(
        Kokkos::WithoutInitializing, Kokkos::DefaultHostExecutionSpace{}, val);

    Kokkos::parallel_for("CopyRowPtr",
                         Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                             0, rowptr.extent(0)),
                         [=](int i) { rowptr_host(i) = rowptr_data[i]; });
    Kokkos::parallel_for("CopyColVal",
                         Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
                             0, col.extent(0)),
                         [=](int i) {
                             col_host(i) = col_data[i];
                             val_host(i) = val_data[i];
                         });

    // Copy to device
    Kokkos::deep_copy(rowptr, rowptr_host);
    Kokkos::deep_copy(col, col_host);
    Kokkos::deep_copy(val, val_host);

    const auto nrows = rowptr_data.size() - 1;
    const auto ncols =
        col_data.size() > 0
            ? *std::max_element(col_data.begin(), col_data.end()) + 1
            : 0;
    const auto nnz = val_data.size();

    CSRMatrixType Acsr("A", nrows, ncols, nnz, val, rowptr, col);

    return Acsr;
}

template <typename Execspace>
Kokkos::View<double*, typename Execspace::memory_space> build_jacobi(
    const KokkosSparse::CrsMatrix<double, int, Execspace, void, int>& A) {
    using MemorySpace = typename Execspace::memory_space;
    Kokkos::View<double*, MemorySpace> inv_diag("inv_diag", A.numRows());

    auto rowmap = A.graph.row_map;
    auto entries = A.graph.entries;
    auto values = A.values;

    Kokkos::parallel_for(
        "Jacobi", Kokkos::RangePolicy<Execspace>(0, A.numRows()),
        KOKKOS_LAMBDA(const int i) {
            for (int k = rowmap(i); k < rowmap(i + 1); k++) {
                if (entries(k) == i) {
                    inv_diag(i) = double(1.0) / values(k);
                    break;
                }
            }
        });

    return inv_diag;
}

template <typename Execspace>
void apply_jacobi(
    const Kokkos::View<double*, typename Execspace::memory_space> inv_diag,
    const Kokkos::View<double*, typename Execspace::memory_space> r,
    Kokkos::View<double*, typename Execspace::memory_space> z) {
    Kokkos::parallel_for(
        "ApplyJacobi", Kokkos::RangePolicy<Execspace>(0, r.extent(0)),
        KOKKOS_LAMBDA(const int i) { z(i) = inv_diag(i) * r(i); });
}

template <typename Execspace>
int pcg_jacobi(int max_iter, double tol) {
    using MemorySpace = typename Execspace::memory_space;

    // Read CSV files and create Kokkos views for rowptr, col, val CSR
    auto A = read_csv_files<Execspace>();

    // Create vector `b` = ones(N), initial guess `x0` = zeros(N)
    const int N = A.numRows();
    auto b = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(Execspace{}, Kokkos::WithoutInitializing, "b"), N);
    auto x = Kokkos::View<double*, MemorySpace>(
        Kokkos::view_alloc(Execspace{}, Kokkos::WithoutInitializing, "x0"), N);

    Kokkos::deep_copy(b, 1.0);
    Kokkos::deep_copy(x, 0.0);

    const int n = A.numRows();

    Kokkos::View<double*, MemorySpace> r("r", n);
    Kokkos::View<double*, MemorySpace> z("z", n);
    Kokkos::View<double*, MemorySpace> p("p", n);
    Kokkos::View<double*, MemorySpace> Ap("Ap", n);

    // r = b - A x
    KokkosSparse::spmv("N", 1.0, A, x, 0.0, r);
    KokkosBlas::axpby(1.0, b, -1.0, r);

    auto inv_diag = build_jacobi<Execspace>(A);

    apply_jacobi<Execspace>(inv_diag, r, z);

    Kokkos::deep_copy(p, z);

    double rz_old = KokkosBlas::dot(r, z);
    double rnorm0 = KokkosBlas::nrm2(b);
    // double rnorm0 = KokkosBlas::nrm2(r);

    int iter = 0;

    printf("Iteration %d:\n  relative error = %.6e\n", iter, rz_old);

    for (iter = 0; iter < max_iter; iter++) {
        // Ap = A p
        KokkosSparse::spmv("N", 1.0, A, p, 0.0, Ap);

        double alpha = rz_old / KokkosBlas::dot(p, Ap);

        // x = x + alpha p
        KokkosBlas::axpy(alpha, p, x);

        // r = r - alpha Ap
        KokkosBlas::axpy(-alpha, Ap, r);

        double rnorm = KokkosBlas::nrm2(r);
        if ((iter + 1) % 10 == 0) {
            printf("Iteration %d:\n  relative error ||r||/||b||= %.6e\n",
                   iter + 1, rnorm / rnorm0);
        }
        if (rnorm / rnorm0 < tol) {
            printf("Iteration %d:\n  relative error ||r||/||b||= %.6e\n",
                   iter + 1, rnorm / rnorm0);
            return iter + 1;
        }

        apply_jacobi<Execspace>(inv_diag, r, z);

        double rz_new = KokkosBlas::dot(r, z);
        double beta = rz_new / rz_old;

        // p = z + beta p
        KokkosBlas::axpby(1.0, z, beta, p);

        rz_old = rz_new;
    }
    printf("Iteration %d:\n  relative error = %.6e\n", iter + 1, rz_old);
    return std::min(iter, max_iter);
}
