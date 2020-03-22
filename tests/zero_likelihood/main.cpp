#include <pyprob_cpp.h>
#include <string>

// Gaussian with unkown mean
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

xt::xarray<double> forward()
{
  auto prior = pyprob_cpp::distributions::Uniform(0, 1);

  for(int i = 0; i < 5; i++) {
    auto x = pyprob_cpp::sample(prior);
    auto likelihood = pyprob_cpp::distributions::Uniform(x, 1);
    pyprob_cpp::observe(likelihood, "obs_" + std::to_string(i));
  }

  return 0;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "ipc://@my_test";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Model with occasional zero likelihood");
  model.startServer(serverAddress);
  return 0;
}