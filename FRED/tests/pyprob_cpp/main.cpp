#include <pyprob_cpp.h>

// Gaussian with unkown mean
// http://www.robots.ox.ac.uk/~fwood/assets/pdf/Wood-AISTATS-2014.pdf

xt::xarray<double> forward()
{
  /*The simulator goes here*/
  /*
    The simulator code should be modified such that all the calls to
    random number generators are replaced with pyprob_cpp::sample(...).
    This function takes a distribution from pyprob_cpp::distributions
    (and optionally a name for the random variable being sampled).
    There is also pyprob_cpp::observe statements which are used for
    inference.
  */
  auto prior_mean = 1;
  auto prior_stddev = std::sqrt(5);
  auto likelihood_stddev = std::sqrt(2);

  auto prior = pyprob_cpp::distributions::Normal(prior_mean, prior_stddev);
  auto mu = pyprob_cpp::sample(prior);

  auto likelihood = pyprob_cpp::distributions::Normal(mu, likelihood_stddev);
  pyprob_cpp::observe(likelihood, "obs");

  return mu;
}


int main(int argc, char *argv[])
{
  // Extract the inter-process communication address from the arguments.
  auto serverAddress = (argc > 1) ? argv[1] : "ipc://@my_test";
  // Instantiate the model object
  pyprob_cpp::Model model = pyprob_cpp::Model(forward, "Gaussian with unknown mean C++");
  // Start running the simulator
  model.startServer(serverAddress);
  return 0;
}