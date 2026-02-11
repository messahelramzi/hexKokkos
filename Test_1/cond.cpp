#include <CLI/CLI.hpp>

#include "cond.hxx"

int main(int argc, char* argv[]) {
    // Parse command line arguments with CLI11
    CLI::App app{"cond - condition number of a matrix"};
    std::string execution_space = "device";
    std::string matrix_file = "A.csv";
    std::string vector_file = "b.csv";
    std::string view_layout = "none";
    std::string view_layout_in = "none";
    double lambda_max = std::numeric_limits<double>::min();
    double lambda_min = std::numeric_limits<double>::max();
    int iter = -1; 

    app.add_option("--exec_space", execution_space,
                   "Execution space (device|host)")
        ->default_val("device");
    app.add_option("--matrix", matrix_file, "Path to matrix CSV file")
        ->default_val("A.csv");
    app.add_option("--vector", vector_file, "Path to vector CSV file")
        ->default_val("b.csv");
    app.add_option("--view_layout", view_layout_in, "Default view layout")
        ->default_val("none");

    CLI11_PARSE(app, argc, argv);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        using default_view_t = Kokkos::View<double**>;
        using default_layout_t = typename default_view_t::array_layout;
        if constexpr (std::is_same_v<default_layout_t, Kokkos::LayoutLeft>)
            view_layout = "LayoutLeft";
        else if constexpr (std::is_same_v<default_layout_t,
                                          Kokkos::LayoutRight>)
            view_layout = "LayoutRight";
        else
            std::cout << "Default layout: other\n";

        if (view_layout_in != "none") {
            if (view_layout_in == "LayoutLeft" ||
                view_layout_in == "LayoutRight") {
                view_layout = view_layout_in;
            } else {
                std::cerr << "Invalid view layout specified: " << view_layout_in
                          << ". Using default layout: " << view_layout
                          << std::endl;
            }
        }

        std::cout << "=====================================" << std::endl;
        std::cout << "Execution space: " << execution_space << std::endl;
        std::cout << "Matrix file: " << matrix_file << std::endl;
        std::cout << "Vector file: " << vector_file << std::endl;
        std::cout << "Default view layout: " << view_layout << std::endl;
        std::cout << "=====================================" << std::endl;
        std::cout << std::endl;

        double elapsed = std::numeric_limits<double>::max();

        if (execution_space == "device") {
            if (view_layout == "LayoutLeft") {
                auto [A, b] =
                    read_csv_files<ExecDeviceSpace, Kokkos::LayoutLeft>("A.csv",
                                                                  "b.csv");
                Kokkos::Timer timer;
                std::tie(lambda_max, iter) = power_iteration(A, b);
                std::tie(lambda_min, iter) = inverse_power_iteration(A, b);
                elapsed = timer.seconds();
            } else {
                auto [A, b] =
                    read_csv_files<ExecDeviceSpace, Kokkos::LayoutRight>("A.csv",
                                                                   "b.csv");
                Kokkos::Timer timer;
                std::tie(lambda_max, iter) = power_iteration(A, b);
                std::tie(lambda_min, iter) = inverse_power_iteration(A, b);
                elapsed = timer.seconds();
            }
        } else {
            if (view_layout == "LayoutLeft") {
                auto [A, b] =
                    read_csv_files<ExecHostSpace, Kokkos::LayoutLeft>("A.csv",
                                                                      "b.csv");
                Kokkos::Timer timer;
                std::tie(lambda_max, iter) = power_iteration(A, b);
                std::tie(lambda_min, iter) = inverse_power_iteration(A, b);
                elapsed = timer.seconds();
            } else {
                auto [A, b] =
                    read_csv_files<ExecHostSpace, Kokkos::LayoutRight>("A.csv",
                                                                       "b.csv");
                Kokkos::Timer timer;
                std::tie(lambda_max, iter) = power_iteration(A, b);
                std::tie(lambda_min, iter) = inverse_power_iteration(A, b);
                elapsed = timer.seconds();
            }
        }
        Kokkos::fence();
        std::cout << "Estimated largest eigenvalue: " << lambda_max << std::endl;
        std::cout << "Estimated smallest eigenvalue: " << lambda_min << std::endl;
        std::cout << "Estimated condition number: " << lambda_max / lambda_min
                  << std::endl;
        std::cout << "elapsed time: " << elapsed << " seconds" << std::endl;
    }  // End of Kokkos scope
    Kokkos::finalize();

    return 0;
}
