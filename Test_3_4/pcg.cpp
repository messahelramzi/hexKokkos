#include <CLI/CLI.hpp>

#include "pcg.hxx"

int main(int argc, char* argv[]) {
    // Parse command line arguments with CLI11
    CLI::App app{"pcg - preconditioned conjugate gradient"};
    std::string execution_space = "device";

    double tol = 1e-8;
    int max_iter = 500;
    app.add_option("--exec_space", execution_space,
                   "Execution space (device|host)")
        ->default_val("device");
    app.add_option("--tol", tol, "covnrgence tolerance")->default_val(1e-8);
    app.add_option("--max_iter", max_iter, "Maximum number of iterations")
        ->default_val(500);

    CLI11_PARSE(app, argc, argv);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        // Kokkos::print_configuration(std::cout);

        std::cout << "=====================================" << std::endl;
        std::cout << "Execution space: " << execution_space << std::endl;
        std::cout << "tol: " << tol << std::endl;
        std::cout << "max_iter: " << max_iter << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << std::endl;

        int iters = 0;
        double elapsed = 0.0;
        if (execution_space == "device") {
            Kokkos::Timer t;
            iters = pcg_jacobi<ExecDeviceSpace>(max_iter, tol);
            elapsed = t.seconds();
        } else {
            Kokkos::Timer t;
            iters = pcg_jacobi<ExecHostSpace>(max_iter, tol);
            elapsed = t.seconds();
        }
        if (iters < max_iter) {
            printf("PCG converged in %d iterations\n", iters);
        } else {
            printf("PCG did not converge in %d iterations\n", max_iter);
        }
        std::cout << std::endl;
        printf("Execution time: %.6f seconds\n", elapsed);
        printf("Time-per-iteration: %.6f seconds\n", elapsed / iters);
    }  // End of Kokkos scope
    Kokkos::finalize();

    return 0;
}
