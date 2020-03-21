#include <pyprob_cpp.h>

// Gaussian with unkown mean
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

xt::xarray<double> forward()
{
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  auto prior = pyprob_cpp::distributions::Normal(prior_mean, prior_stddev);
  auto mu = pyprob_cpp::sample(prior);

  auto likelihood = pyprob_cpp::distributions::Normal(mu, likelihood_stddev);

  return mu;
}


int main(int argc, char *argv[])
{
  auto serverAddress = (argc > 1) ? argv[1] : "ipc://@my_test";
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Gaussian with unknown mean C++");
  model.startServer(serverAddress);
  return 0;
}